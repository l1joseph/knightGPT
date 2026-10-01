"""Base tool interface for KnightGPT agent system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..api.request_context import RequestContext


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_context(self, max_chars: int = 2000) -> str:
        """Format result as context string for LLM consumption."""
        if not self.success:
            return f"[{self.tool_name} ERROR]: {self.error}"
        text = str(self.data)
        if len(text) > max_chars:
            text = text[:max_chars] + f"... (truncated, {len(text)} total chars)"
        return f"[{self.tool_name}]: {text}"


class BaseTool(ABC):
    """Base class for all domain tools."""

    name: str = "base"
    description: str = "Base tool"

    @abstractmethod
    def execute(
        self,
        query: str,
        *,
        request_context: "RequestContext | None" = None,
        **kwargs,
    ) -> ToolResult:
        """Execute the tool with the given query.

        request_context is injected by AgentOrchestrator.run()'s dispatch
        loop -- never parsed from the model's JSON tool-call arguments
        (see src/agents/orchestrator.py). Most tools ignore it entirely
        via **kwargs; only ingest_paper and search_corpus read it.
        """
        ...

    @property
    def schema(self) -> dict:
        """JSON schema for tool parameters (for LLM function calling)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        }

    @property
    def openai_tool_schema(self) -> dict:
        """This tool's .schema wrapped in the {type, function} envelope
        OpenAI's tools=[...] function-calling parameter requires."""
        return {
            "type": "function",
            "function": self.schema,
        }
