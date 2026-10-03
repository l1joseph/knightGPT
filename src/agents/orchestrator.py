"""Multi-agent orchestrator for KnightGPT (Eubiota-inspired).

Runs a real OpenAI-style function-calling loop: the model is given
tools=[...] and decides whether/which tools to call; results are fed back
as tool-role messages until the model returns a final answer or
max_tool_rounds is hit. Emits structured, transport-agnostic lifecycle
events via an optional on_event callback -- see
docs/superpowers/specs/2026-09-14-knightgpt-webui-deploy-design.md for why
(modeled on Stanford's Eubiota project's on_event/_emit pattern): this
class knows nothing about SSE or OpenAI's chat.completion.chunk wire
format, which stays in src/api/sse_adapter.py.
"""

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, TYPE_CHECKING

from openai import OpenAI

from ..retrieval import BaseRetriever
from ..tools.base import BaseTool, ToolResult
from ..tools.pubmed import PubMedTool
from ..tools.openalex import OpenAlexTool
from ..tools.kegg import KEGGTool
from ..tools.qiime2 import QIIME2Tool
from ..tools.ingest_paper import IngestPaperTool
from ..tools.search_corpus import SearchCorpusTool
from ..utils import get_logger, get_settings

if TYPE_CHECKING:
    # Deferred to a TYPE_CHECKING-only import (same pattern as
    # src/tools/base.py) because src.api's package __init__ imports
    # src.api.main, which imports AgentOrchestrator from this very module
    # -- a module-level `from ..api.request_context import RequestContext`
    # here would make `import src.agents` raise ImportError: cannot import
    # name 'AgentOrchestrator' from partially initialized module
    # 'src.agents' (circular import). The real import happens lazily
    # inside run(), by which point both packages are fully initialized.
    from ..api.request_context import RequestContext

logger = get_logger(__name__)
settings = get_settings()

GENERATOR_SYSTEM_PROMPT = """You are a microbiome research assistant. Use the
available tools when they would help answer the question, then generate a
comprehensive final answer using ONLY the provided tool results and any
knowledge graph context.

Rules:
- Cite sources using [Source: filename] or [DOI: xxx] format
- If the context doesn't contain enough information, say so
- Be precise about methods and findings
- Distinguish between established knowledge and recent findings"""


@dataclass
class AgentContext:
    """Accumulated context from a run() call."""

    original_query: str = ""
    tool_results: list[ToolResult] = field(default_factory=list)
    rag_context: str = ""
    final_answer: str = ""


class AgentOrchestrator:
    """
    Coordinates a real function-calling agent loop.

    Usage:
        orchestrator = AgentOrchestrator(retriever=retriever)
        result = orchestrator.run("What role does Prevotella play in gut health?")
        print(result.final_answer)
    """

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
    ):
        self.retriever = retriever

        self.tools: dict[str, BaseTool] = {
            "pubmed_search": PubMedTool(),
            "openalex_search": OpenAlexTool(),
            "kegg_lookup": KEGGTool(),
            "qiime2_docs": QIIME2Tool(),
            "ingest_paper": IngestPaperTool(retriever=retriever),
            "search_corpus": SearchCorpusTool(retriever=retriever),
        }

        self.client = OpenAI(
            api_key=settings.vllm.api_key,
            base_url=settings.vllm.inference_url,
        )
        self.model = settings.vllm.inference_model

    def run(
        self,
        query: str,
        top_k: int = 5,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        max_tool_rounds: int = 5,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        history: list[dict] | None = None,
        request_context: "RequestContext | None" = None,
    ) -> AgentContext:
        """Run the function-calling agent loop.

        Args:
            query: the user's question.
            top_k: RAG retrieval depth (used only if self.retriever is set).
            on_event: optional callback fired synchronously for every
                lifecycle event ({"type": "tool_call"|"tool_result"|
                "token"|"error"|"done", ...} -- see src/api/sse_adapter.py
                for the exact shapes consumed downstream).
            max_tool_rounds: safety cap on tool-calling rounds; if hit, one
                final answer is forced with no further tools offered.
            temperature: sampling temperature for every LLM call this run
                makes (both the per-round tool-calling calls and the forced
                final-answer call).
            max_tokens: max tokens for every LLM call this run makes.
            history: prior turns of this conversation, as OpenAI-style
                {"role": "user"|"assistant", "content": str} dicts, oldest
                first, NOT including the current query -- confirmed live on
                kl-remote that omitting this made every follow-up question
                in a multi-turn Open WebUI chat ("so what are the
                microbes" after "what microbes are associated with AD?")
                get answered as if it were a brand-new conversation.
                Passed straight through to every LLM call this run makes.
                Optional; omitted or empty means a single-turn conversation
                (unchanged prior behavior).
            request_context: identity + collection scope for this request
                (see src/api/request_context.py), injected into every
                tool.execute() call as a keyword-only argument the
                model's JSON tool-call arguments can never populate or
                override. Optional; omitted (the default) uses an
                all-None/non-admin/no-collection context, preserving
                existing behavior for any caller that doesn't pass one
                (e.g. a script calling run() directly).
        """
        # Imported lazily (not at module level) to avoid a circular import
        # -- see the TYPE_CHECKING comment near the top of this file.
        from ..api.request_context import RequestContext

        emit = on_event or (lambda event: None)
        ctx = AgentContext(original_query=query)
        effective_request_context = request_context or RequestContext()

        rag_context = self._retrieve_rag_context(query, top_k)
        ctx.rag_context = rag_context

        messages: list[dict] = [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        ]
        if rag_context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Knowledge graph context:\n{rag_context}",
                }
            )
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        tool_schemas = [tool.openai_tool_schema for tool in self.tools.values()]

        for _round_num in range(max_tool_rounds):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_schemas,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                content, tool_calls = self._consume_stream(stream, emit)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                ctx.final_answer = f"The language model request failed: {e}"
                emit({"type": "error", "message": str(e)})
                emit({"type": "done"})
                return ctx

            if not tool_calls:
                # content was already streamed out token-by-token above --
                # don't re-emit it as one more giant token event.
                ctx.final_answer = content
                emit({"type": "done"})
                return ctx

            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for call_index, tc in enumerate(tool_calls):
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                emit(
                    {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "args": args,
                        "call_id": tc.id,
                        "index": call_index,
                    }
                )

                tool = self.tools.get(tool_name)
                if tool is None:
                    result = ToolResult(
                        tool_name=tool_name,
                        success=False,
                        error=f"Unknown tool: {tool_name}",
                    )
                else:
                    result = tool.execute(
                        args.get("query", ""),
                        request_context=effective_request_context,
                        **{
                            k: v
                            for k, v in args.items()
                            if k not in ("query", "request_context")
                        },
                    )
                ctx.tool_results.append(result)

                emit(
                    {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "call_id": tc.id,
                        "success": result.success,
                        "summary": result.to_context(max_chars=500),
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.to_context(),
                    }
                )

        # max_tool_rounds exhausted without a final answer -- force one,
        # with no tools offered so the model cannot request yet another round.
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": "Provide your best answer now with the information gathered so far.",
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            content, _tool_calls = self._consume_stream(stream, emit)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            ctx.final_answer = f"The language model request failed: {e}"
            emit({"type": "error", "message": str(e)})
            emit({"type": "done"})
            return ctx

        # No tools= was offered on this call, so content (already streamed
        # out token-by-token above) is necessarily the final answer.
        ctx.final_answer = content
        emit({"type": "done"})
        return ctx

    def _consume_stream(
        self,
        stream: Any,
        emit: Callable[[dict[str, Any]], None],
    ) -> tuple[str, list[SimpleNamespace] | None]:
        """Iterate a `stream=True` chat.completions.create() response.

        Emits a `token` event for each content fragment AS IT ARRIVES (real
        token-by-token streaming), and accumulates any tool-call fragments
        (keyed by their `index`, since multiple tool calls can stream in
        parallel) into complete tool-call records. A given streaming turn
        is either content-only or tool-calls-only in practice, never mixed.

        Tool-call `function.arguments` fragments are concatenated as plain
        strings and are NEVER parsed as JSON here -- only the caller, after
        the stream has fully ended, should attempt json.loads() on the
        concatenated result. Parsing a partial fragment mid-stream would
        blow up on every turn since the JSON is only valid once complete.

        Returns:
            (content, tool_calls) where content is the full accumulated
            answer text (already emitted piecemeal via on_event) and
            tool_calls is None if no tool-call fragments were seen, or a
            list of SimpleNamespace objects -- ordered by index -- shaped
            like the OpenAI SDK's ChatCompletionMessageToolCall (`.id`,
            `.function.name`, `.function.arguments`) so existing
            downstream code that reads those via attribute access keeps
            working unchanged.
        """
        content_parts: list[str] = []
        tool_call_frags: dict[int, dict[str, str | None]] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                for tc_delta in delta_tool_calls:
                    frag = tool_call_frags.setdefault(
                        tc_delta.index,
                        {"id": None, "name": None, "arguments": ""},
                    )
                    if tc_delta.id:
                        frag["id"] = tc_delta.id
                    function = getattr(tc_delta, "function", None)
                    if function is not None:
                        if function.name:
                            frag["name"] = function.name
                        if function.arguments:
                            frag["arguments"] += function.arguments
                # No `continue` here: a chunk is not guaranteed to carry
                # only one of content/tool_calls. In practice a turn is
                # either content-only or tool-calls-only, so this branch
                # and the content check below are usually mutually
                # exclusive per chunk anyway -- but checking both costs
                # nothing and means a chunk that happens to carry both
                # (e.g. a model emitting a short preamble alongside a tool
                # call) never silently loses its content fragment.

            content_fragment = getattr(delta, "content", None)
            if content_fragment:
                content_parts.append(content_fragment)
                emit({"type": "token", "content": content_fragment})

        content = "".join(content_parts)

        if not tool_call_frags:
            return content, None

        tool_calls = [
            SimpleNamespace(
                id=frag["id"],
                function=SimpleNamespace(
                    name=frag["name"], arguments=frag["arguments"]
                ),
            )
            for _, frag in sorted(tool_call_frags.items())
        ]
        return content, tool_calls

    def _retrieve_rag_context(self, query: str, top_k: int) -> str:
        if not self.retriever:
            return ""
        try:
            retrieval = self.retriever.retrieve(
                query=query, top_k=top_k, expand_context=True
            )
            return self.retriever.format_context(retrieval.chunks)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return ""
