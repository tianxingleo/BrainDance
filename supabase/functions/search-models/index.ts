/**
 * BrainDance 语义搜索 Edge Function
 *
 * 当前文件只保留 HTTP 入口职责。
 * 检索主逻辑已经下沉到 `shared.ts`，供 `search-models` 与 `agent-recall`
 * 共享，避免时间解析、Embedding 和向量检索逻辑继续分叉。
 */

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { runSearchModelsQuery } from "./shared.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

function errorResponse(message: string, status = 500): Response {
  console.error(`[Search] 错误: ${message}`);
  return new Response(
    JSON.stringify({
      success: false,
      error: message,
    }),
    {
      status,
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json",
      },
    },
  );
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const { query, threshold = 0.5 } = await req.json();

    if (!query || typeof query !== "string") {
      return errorResponse("缺少或无效的搜索关键词 'query'", 400);
    }
    if (query.trim().length === 0) {
      return errorResponse("搜索关键词不能为空", 400);
    }
    if (query.length > 500) {
      return errorResponse("搜索关键词过长（最大 500 字符）", 400);
    }

    const result = await runSearchModelsQuery(query, threshold);
    return new Response(JSON.stringify(result), {
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json",
      },
    });
  } catch (e) {
    return errorResponse(e instanceof Error ? e.message : String(e), 500);
  }
});

console.log("[Search] Edge Function 已初始化完成，等待请求...");
