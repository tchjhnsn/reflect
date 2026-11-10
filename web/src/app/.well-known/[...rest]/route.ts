import { oauthProviderAuthServerMetadata } from "@better-auth/oauth-provider";
import { auth } from "@/server/auth";

// OAuth discovery for MCP clients, at the root where they look for it: the authorization server's
// metadata and the MCP endpoint's protected resource metadata (RFC 8414, RFC 9728). Better Auth answers
// both; the auth server's paths live under /api/auth, so this passes the request on.
const serverMetadata = oauthProviderAuthServerMetadata(auth);

export async function GET(request: Request): Promise<Response> {
  const path = new URL(request.url).pathname;
  if (path.startsWith("/.well-known/oauth-authorization-server") || path.startsWith("/.well-known/openid-configuration")) return serverMetadata(request);
  if (path.startsWith("/.well-known/oauth-protected-resource")) return auth.handler(request);
  return new Response("not found", { status: 404 });
}
