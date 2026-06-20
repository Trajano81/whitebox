"""MCP server for whitebox.

The tool logic lives in `tools.py` (pure functions over a Session, no MCP
dependency, fully unit-testable). `server.py` wires those tools to FastMCP
(stdio). Run with: python -m whitebox.mcp.server
"""
from .tools import Session, ToolError

__all__ = ["Session", "ToolError"]
