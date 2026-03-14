"""Base tool interface for KnightGPT agent system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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
    def execute(self, query: str, **kwargs) -> ToolResult:
        """Execute the tool with the given query."""
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
