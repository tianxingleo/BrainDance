import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import {
  type AgentProgressEvent,
  runSpatialSearchAgent,
} from "../_shared/agent-core/spatialAgent.ts";
import { agentRecallRequestSchema } from "./schemas/request.ts";

export const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

function errorResponse(message: string, status = 500): Response {
  console.error(`[AgentRecall] 错误: ${message}`);
  return new Response(
    JSON.stringify({ success: false, error: message }),
    {
      status,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    },
  );
}

const encoder = new TextEncoder();

function isStreamingRequest(req: Request): boolean {
  const accept = req.headers.get("accept") ?? "";
  const streamFlag = new URL(req.url).searchParams.get("stream");
  return streamFlag === "1" || accept.includes("application/x-ndjson");
}

function writeNdjsonLine(
  controller: ReadableStreamDefaultController<Uint8Array>,
  event: string,
  data: unknown,
): void {
  controller.enqueue(
    encoder.encode(`${JSON.stringify({ event, data })}\n`),
  );
}

function chunkText(text: string, chunkSize = 28): string[] {
  const normalized = text.trim();
  if (!normalized) return [];

  const chunks: string[] = [];
  for (let index = 0; index < normalized.length; index += chunkSize) {
    chunks.push(normalized.slice(index, index + chunkSize));
  }
  return chunks;
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const body = await req.json();
    const parsed = agentRecallRequestSchema.safeParse(body);
    if (!parsed.success) {
      return errorResponse(
        parsed.error.issues[0]?.message ?? "请求参数无效",
        400,
      );
    }

    const agentOptions = {
      selectedModelIds: parsed.data.selectedModelIds,
      executionMode: parsed.data.executionMode,
      currentSceneId: parsed.data.currentSceneId,
      currentModelId: parsed.data.currentModelId,
      currentMode: parsed.data.currentMode,
      candidateSceneIds: parsed.data.candidateSceneIds,
      sessionId: parsed.data.sessionId,
      conversationSummary: parsed.data.conversationSummary,
      sessionState: parsed.data.sessionState,
    };

    if (isStreamingRequest(req)) {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          void (async () => {
            try {
              const result = await runSpatialSearchAgent(
                parsed.data.query,
                agentOptions,
                {
                  onEvent: async (event: AgentProgressEvent) => {
                    writeNdjsonLine(controller, event.event, event.data);
                  },
                },
              );

              for (const chunk of chunkText(result.answer)) {
                writeNdjsonLine(controller, "message", { delta: chunk });
              }

              writeNdjsonLine(controller, "done", result);
            } catch (error) {
              writeNdjsonLine(controller, "error", {
                message: error instanceof Error ? error.message : String(error),
              });
            } finally {
              controller.close();
            }
          })();
        },
      });

      return new Response(stream, {
        headers: {
          ...corsHeaders,
          "Content-Type": "application/x-ndjson; charset=utf-8",
          "Cache-Control": "no-cache, no-transform",
          "X-Accel-Buffering": "no",
        },
      });
    }

    const result = await runSpatialSearchAgent(parsed.data.query, agentOptions);

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    return errorResponse(
      error instanceof Error ? error.message : String(error),
      500,
    );
  }
});

console.log("[AgentRecall] Edge Function 已初始化完成，等待请求...");
