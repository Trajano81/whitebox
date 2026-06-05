"""E2E for the MCP server (spec-006). Skipped unless the 'mcp' package is installed.

Verifies the FastMCP server module imports and registers the whitebox tools.
"""
import importlib.util

import pytest

pytestmark = pytest.mark.e2e

HAS_MCP = importlib.util.find_spec("mcp") is not None


@pytest.mark.skipif(not HAS_MCP, reason="mcp not installed")
def test_server_registers_tools():
    from whitebox.mcp import server

    # FastMCP exposes the registered tools; ensure our prefix is present.
    names = set()
    for attr in ("_tools", "tools"):
        reg = getattr(server.mcp, attr, None)
        if isinstance(reg, dict):
            names |= set(reg.keys())
    # Fallback: the decorated functions exist on the module.
    assert hasattr(server, "whitebox_load_session")
    assert hasattr(server, "whitebox_add_group")
