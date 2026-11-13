import { gateway } from "@ai-sdk/gateway";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { LanguageModel } from "ai";

/**
 * The chat's model: gpt-oss-120b through Vercel AI Gateway when deployed (it signs in with the
 * deployment's OIDC token), otherwise the same model on Ollama Cloud. Null when neither is set up.
 */
export function chatModel(env: Record<string, string | undefined> = process.env): LanguageModel | null {
  if (env.VERCEL || env.AI_GATEWAY_API_KEY) return gateway("openai/gpt-oss-120b");
  if (!env.OLLAMA_API_KEY) return null;
  const ollama = createOpenAICompatible({ name: "ollama", baseURL: `${env.OLLAMA_HOST ?? "https://ollama.com"}/v1`, apiKey: env.OLLAMA_API_KEY });
  return ollama.chatModel(env.REFLECT_READER_MODEL ?? "gpt-oss:120b");
}
