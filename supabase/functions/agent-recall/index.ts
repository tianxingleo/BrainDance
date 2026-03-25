import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { runRecallAgent } from "./agent/recallAgent.ts";
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

    const result = await runRecallAgent(parsed.data.query);
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
