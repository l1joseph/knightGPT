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
from typing import Any, Callable

from openai import OpenAI

from ..retrieval import BaseRetriever
from ..tools.base import BaseTool, ToolResult
from ..tools.pubmed import PubMedTool
from ..tools.openalex import OpenAlexTool
from ..tools.kegg import KEGGTool
from ..tools.qiime2 import QIIME2Tool
from ..utils import get_logger, get_settings

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
    ) -> AgentContext:
        """Run the function-calling agent loop.

        Args:
            query: the user's question.
            top_k: RAG retrieval depth (used only if self.retriever is set).
            on_event: optional callback fired synchronously for every
                lifecycle event ({"type": "tool_call"|"tool_result"|
                "token"|"done", ...} -- see src/api/sse_adapter.py for the
                exact shapes consumed downstream).
            max_tool_rounds: safety cap on tool-calling rounds; if hit, one
                final answer is forced with no further tools offered.
        """
        emit = on_event or (lambda event: None)
        ctx = AgentContext(original_query=query)

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
        messages.append({"role": "user", "content": query})

        tool_schemas = [tool.openai_tool_schema for tool in self.tools.values()]

        for _round_num in range(max_tool_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_schemas,
                temperature=0.3,
                max_tokens=2000,
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)

            if not tool_calls:
                answer = choice.message.content or ""
                ctx.final_answer = answer
                emit({"type": "token", "content": answer})
                emit({"type": "done"})
                return ctx

            messages.append(
                {
                    "role": "assistant",
                    "content": choice.message.content,
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

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                emit(
                    {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "args": args,
                        "call_id": tc.id,
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
                        **{k: v for k, v in args.items() if k != "query"},
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
            + [
                {
                    "role": "user",
                    "content": "Provide your best answer now with the information gathered so far.",
                }
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        answer = response.choices[0].message.content or ""
        ctx.final_answer = answer
        emit({"type": "token", "content": answer})
        emit({"type": "done"})
        return ctx

    def _retrieve_rag_context(self, query: str, top_k: int) -> str:
        if not self.retriever:
            return ""
        try:
            retrieval = self.retriever.retrieve(
                query=query, top_k=top_k, expand_context=True
            )
            return self.retriever.format_context(
                retrieval.chunks, retrieval.similarity_scores
            )
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return ""
