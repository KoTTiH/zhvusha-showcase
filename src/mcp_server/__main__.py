"""Entry point for Zhvusha knowledge MCP server.

Usage:
    python -m src.mcp_server              # stdio transport
    python -m src.mcp_server --http       # HTTP transport + dashboard
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()  # Ensure DATABASE_URL is available when spawned by an agent client.

from src.mcp_server.server import mcp, register_dashboard_routes  # noqa: E402

if __name__ == "__main__":
    if "--http" in sys.argv:
        from mcp.server.transport_security import TransportSecuritySettings

        register_dashboard_routes()
        mcp.settings.port = 8765
        # Disable host validation — server is behind a tunnel (cloudflare/ngrok)
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        )
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
