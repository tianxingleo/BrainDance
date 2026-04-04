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
  return streamFlag === "1" ||
    accept.includes("application/x-ndjson") ||
    accept.includes("text/event-stream");
}

function prefersSse(req: Request): boolean {
  const accept = req.headers.get("accept") ?? "";
  return accept.includes("text/event-stream");
}

function writeStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  format: "ndjson" | "sse",
  event: string,
  data: unknown,
): void {
  const payload = format === "sse"
    ? `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
    : `${JSON.stringify({ event, data })}\n`;
  controller.enqueue(encoder.encode(payload));
  console.log(`[AgentStream] Enqueued event: ${event}`);
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
      const streamFormat = prefersSse(req) ? "sse" : "ndjson";
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          // 发送一条占位数据以快速撑破 Nginx 等反向代理的缓冲区 (约2KB)
          writeStreamEvent(controller, streamFormat, "ping", {
            message: " ".repeat(2048),
          });
          writeStreamEvent(controller, streamFormat, "status", {
            phase: "request_received",
            summary: "已收到 Agent 请求，准备进入编排链路",
          });

          void (async () => {
            try {
              const result = await runSpatialSearchAgent(
                parsed.data.query,
                agentOptions,
                {
                  onEvent: async (event: AgentProgressEvent) => {
                    writeStreamEvent(
                      controller,
                      streamFormat,
                      event.event,
                      event.data,
                    );
                  },
                },
              );

              writeStreamEvent(controller, streamFormat, "status", {
                phase: "final_answer",
                summary: "检索与工具调用已完成，正在整理最终回答",
              });
              for (const chunk of chunkText(result.answer)) {
                writeStreamEvent(controller, streamFormat, "message", {
                  delta: chunk,
                });
              }

              writeStreamEvent(controller, streamFormat, "done", result);
            } catch (error) {
              writeStreamEvent(controller, streamFormat, "error", {
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
          "Content-Type": streamFormat === "sse"
            ? "text/event-stream; charset=utf-8"
            : "application/x-ndjson; charset=utf-8",
          "Cache-Control": "no-cache, no-transform",
          "Connection": "keep-alive",
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
