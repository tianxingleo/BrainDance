import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { corsHeaders, runSpatialSearchAgent } from "../_shared/agent-core/spatialAgent.ts";

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
    const { query, selectedModelIds, executionMode, userId } = await req.json();

    if (!query || typeof query !== "string") {
      return errorResponse("缺少或无效的搜索语句 'query'", 400);
    }
    if (query.trim().length === 0) {
      return errorResponse("搜索语句不能为空", 400);
    }
    if (query.length > 500) {
      return errorResponse("搜索语句过长（最大 500 字符）", 400);
    }

    const normalizedSelectedModelIds = Array.isArray(selectedModelIds)
      ? selectedModelIds.filter((item): item is string =>
        typeof item === "string" && item.trim().length > 0
      )
      : undefined;
    const normalizedExecutionMode = executionMode === "execute"
      ? "execute"
      : "preview";

    const result = await runSpatialSearchAgent(query.trim(), {
      selectedModelIds: normalizedSelectedModelIds,
      executionMode: normalizedExecutionMode,
      userId: typeof userId === "string" ? userId : undefined,
    });
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
