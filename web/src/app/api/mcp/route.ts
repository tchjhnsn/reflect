import { requireMcpAuth } from "@better-auth/mcp";
import { createMcpHandler } from "@modelcontextprotocol/server";
import { auth } from "@/server/auth";
import { reflectServer } from "@/server/mcp-tools";
import { mcpResource } from "@/server/site";

// The MCP endpoint a person's own AI connects to. The token names the person (`sub`), and every tool acts
// only for them. Requests are stateless: each one gets a fresh server for its person.
const mcp = createMcpHandler(({ authInfo }) => reflectServer(String(authInfo?.extra?.userId ?? "")), { responseMode: "json" });

const handler = requireMcpAuth(
  auth,
  (request, claims) => {
    const userId = typeof claims.sub === "string" ? claims.sub : "";
    if (!userId) return new Response("token names no person", { status: 401 });
    return mcp.fetch(request, { authInfo: { token: "", clientId: String(claims.azp ?? claims.client_id ?? ""), scopes: String(claims.scope ?? "").split(" "), extra: { userId } } });
  },
  { resource: mcpResource() },
);

export const POST = handler;
export const GET = handler;
