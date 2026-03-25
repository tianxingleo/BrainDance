import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { corsHeaders, runSpatialSearchAgent } from "./agent.ts";

function errorResponse(message: string, status = 500): Response {
  console.error(`[SpatialSearchAgent] 错误: ${message}`);
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
    const { query } = await req.json();

    if (!query || typeof query !== "string") {
      return errorResponse("缺少或无效的搜索语句 'query'", 400);
    }
    if (query.trim().length === 0) {
      return errorResponse("搜索语句不能为空", 400);
    }
    if (query.length > 500) {
      return errorResponse("搜索语句过长（最大 500 字符）", 400);
    }

    const result = await runSpatialSearchAgent(query.trim());
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

console.log("[SpatialSearchAgent] Edge Function 已初始化完成，等待请求...");
