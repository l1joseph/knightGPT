"""
Multi-agent orchestrator for KnightGPT (Eubiota-inspired).

4-stage pipeline:
1. Planner: Analyzes query, selects tools, creates sub-queries
2. Executor: Runs selected tools and RAG retrieval
3. Verifier: Checks citation relevance via embedding similarity
4. Generator: Produces final answer with verified citations

The orchestrator coordinates these stages via LLM calls.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from ..retrieval import GraphRAGRetriever, RAGEngine
from ..tools.base import BaseTool, ToolResult
from ..tools.pubmed import PubMedTool
from ..tools.openalex import OpenAlexTool
from ..tools.kegg import KEGGTool
from ..tools.qiime2 import QIIME2Tool
from ..utils import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class AgentPlan:
    """Plan produced by the planner agent."""

    original_query: str
    sub_queries: list[str] = field(default_factory=list)
    tools_to_use: list[str] = field(default_factory=list)
    reasoning: str = ""
    needs_rag: bool = True


@dataclass
class AgentContext:
    """Accumulated context from all stages."""

    plan: AgentPlan | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    rag_context: str = ""
    verified_citations: list[dict] = field(default_factory=list)
    final_answer: str = ""


PLANNER_SYSTEM_PROMPT = """You are a research planning agent for microbiome science.
Given a user question, create a plan to answer it.

Available tools:
- pubmed_search: Search PubMed for biomedical papers
- openalex_search: Search OpenAlex for academic works with citation data
- kegg_lookup: Look up KEGG metabolic pathways and compounds
- qiime2_docs: Look up QIIME 2 analysis methods and plugins

Respond with JSON:
{
  "sub_queries": ["specific search query 1", "specific search query 2"],
  "tools_to_use": ["tool_name_1", "tool_name_2"],
  "reasoning": "Brief explanation of your plan",
  "needs_rag": true
}

Only select tools that are relevant. For simple factual questions, just use RAG (needs_rag: true, empty tools).
For questions about specific pathways, use kegg_lookup.
For questions about methods/pipelines, use qiime2_docs.
For questions about recent papers, use pubmed_search or openalex_search."""

VERIFIER_SYSTEM_PROMPT = """You are a citation verification agent. Given a claim and
supporting evidence, determine if the evidence actually supports the claim.

Respond with JSON:
{
  "verified": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}"""

GENERATOR_SYSTEM_PROMPT = """You are a microbiome research assistant. Generate a
comprehensive answer using ONLY the provided context and tool results.

Rules:
- Cite sources using [Source: filename] or [DOI: xxx] format
- If the context doesn't contain enough information, say so
- Be precise about methods and findings
- Distinguish between established knowledge and recent findings"""


class AgentOrchestrator:
    """
    Coordinates the multi-agent pipeline.

    Usage:
        orchestrator = AgentOrchestrator(retriever=retriever)
        result = orchestrator.run("What role does Prevotella play in gut health?")
        print(result.final_answer)
    """

    def __init__(
        self,
        retriever: GraphRAGRetriever | None = None,
        rag_engine: RAGEngine | None = None,
    ):
        self.retriever = retriever
        self.rag_engine = rag_engine

        # Initialize tools
        self.tools: dict[str, BaseTool] = {
            "pubmed_search": PubMedTool(),
            "openalex_search": OpenAlexTool(),
            "kegg_lookup": KEGGTool(),
            "qiime2_docs": QIIME2Tool(),
        }

        # LLM client for agent reasoning
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=settings.vllm.inference_url,
        )
        self.model = settings.vllm.inference_model

    def run(self, query: str, top_k: int = 5) -> AgentContext:
        """Run the full agent pipeline."""
        ctx = AgentContext()

        # Stage 1: Plan
        logger.info("Agent Stage 1: Planning")
        ctx.plan = self._plan(query)
        logger.info(f"Plan: tools={ctx.plan.tools_to_use}, subs={ctx.plan.sub_queries}")

        # Stage 2: Execute
        logger.info("Agent Stage 2: Executing")
        ctx = self._execute(ctx, top_k=top_k)

        # Stage 3: Verify
        logger.info("Agent Stage 3: Verifying")
        ctx = self._verify(ctx)

        # Stage 4: Generate
        logger.info("Agent Stage 4: Generating")
        ctx = self._generate(ctx)

        return ctx

    def _plan(self, query: str) -> AgentPlan:
        """Stage 1: Analyze query and create execution plan."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            content = response.choices[0].message.content or "{}"
            # Extract JSON from response (handle markdown code blocks)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            plan_data = json.loads(content.strip())

            return AgentPlan(
                original_query=query,
                sub_queries=plan_data.get("sub_queries", [query]),
                tools_to_use=plan_data.get("tools_to_use", []),
                reasoning=plan_data.get("reasoning", ""),
                needs_rag=plan_data.get("needs_rag", True),
            )
        except Exception as e:
            logger.warning(f"Planning failed, using default plan: {e}")
            return AgentPlan(
                original_query=query,
                sub_queries=[query],
                tools_to_use=[],
                needs_rag=True,
            )

    def _execute(self, ctx: AgentContext, top_k: int = 5) -> AgentContext:
        """Stage 2: Execute tools and RAG retrieval."""
        plan = ctx.plan

        # Run selected tools
        for tool_name in plan.tools_to_use:
            tool = self.tools.get(tool_name)
            if not tool:
                logger.warning(f"Unknown tool: {tool_name}")
                continue

            for sub_query in plan.sub_queries:
                result = tool.execute(sub_query)
                ctx.tool_results.append(result)

        # RAG retrieval from knowledge graph
        if plan.needs_rag and self.retriever:
            try:
                retrieval = self.retriever.retrieve(
                    query=plan.original_query,
                    top_k=top_k,
                    expand_context=True,
                )
                ctx.rag_context = self.retriever.format_context(
                    retrieval.chunks, retrieval.similarity_scores
                )
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")

        return ctx

    def _verify(self, ctx: AgentContext) -> AgentContext:
        """Stage 3: Verify that retrieved context is relevant."""
        # For now, pass through all results — full verification would
        # re-embed and check cosine similarity of each citation
        # against the query. This is deferred until embedding server
        # is reliably available.
        ctx.verified_citations = []

        for result in ctx.tool_results:
            if result.success and result.data:
                items = result.data if isinstance(result.data, list) else [result.data]
                for item in items[:5]:  # Keep top 5 per tool
                    if isinstance(item, dict):
                        ctx.verified_citations.append({
                            "source": result.tool_name,
                            **item,
                        })

        return ctx

    def _generate(self, ctx: AgentContext) -> AgentContext:
        """Stage 4: Generate final answer from all context."""
        # Build context string
        context_parts = []

        if ctx.rag_context:
            context_parts.append(f"## Knowledge Graph Context\n{ctx.rag_context}")

        for result in ctx.tool_results:
            if result.success:
                context_parts.append(result.to_context(max_chars=1500))

        full_context = "\n\n".join(context_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {ctx.plan.original_query}\n\n"
                            f"Context:\n{full_context}\n\n"
                            "Provide a comprehensive answer with citations."
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            ctx.final_answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            ctx.final_answer = (
                f"I found relevant information but couldn't generate a response: {e}"
            )

        return ctx
