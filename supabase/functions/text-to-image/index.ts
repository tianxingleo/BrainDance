/**
 * BrainDance 文本生成图片 Edge Function
 *
 * 功能: 接收用户文本描述，调用 DashScope wanx2.1-t2i-turbo 模型生成图片
 *
 * DashScope 图片生成是异步流程:
 * 1. 提交任务 → 获取 task_id
 * 2. 轮询任务状态 → 直到 SUCCEEDED
 * 3. 返回生成的图片 URL
 *
 * 测试:
 *   curl -X POST http://127.0.0.1:54321/functions/v1/text-to-image \
 *     -H 'Content-Type: application/json' \
 *     -d '{"prompt":"一个红色杯子"}'
 */

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

const DASHSCOPE_IMAGE_SUBMIT_URL =
  "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis";
const DASHSCOPE_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks";

function errorResponse(message: string, status = 500): Response {
  console.error(`[TextToImage] 错误: ${message}`);
  return new Response(
    JSON.stringify({ success: false, error: message }),
    {
      status,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    },
  );
}

/**
 * 提交图片生成任务到 DashScope
 */
async function submitImageTask(
  apiKey: string,
  prompt: string,
): Promise<string> {
  const resp = await fetch(DASHSCOPE_IMAGE_SUBMIT_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "X-DashScope-Async": "enable",
    },
    body: JSON.stringify({
      model: "wanx2.1-t2i-turbo",
      input: { prompt },
      parameters: {
        size: "1024*1024",
        n: 1,
      },
    }),
  });

  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`DashScope 提交任务失败: ${resp.status} - ${err}`);
  }

  const data = await resp.json();
  const taskId = data?.output?.task_id;
  if (!taskId) {
    throw new Error(
      `DashScope 未返回 task_id: ${JSON.stringify(data)}`,
    );
  }

  console.log(`[TextToImage] 任务已提交: ${taskId}`);
  return taskId;
}

/**
 * 轮询任务状态直到完成
 */
async function pollTaskResult(
  apiKey: string,
  taskId: string,
): Promise<string> {
  const maxAttempts = 60; // 最多轮询 60 次 (约 2 分钟)
  const pollInterval = 2000; // 每 2 秒查询一次

  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((resolve) => setTimeout(resolve, pollInterval));

    const resp = await fetch(`${DASHSCOPE_TASK_URL}/${taskId}`, {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
      },
    });

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`DashScope 查询任务失败: ${resp.status} - ${err}`);
    }

    const data = await resp.json();
    const status = data?.output?.task_status;

    console.log(
      `[TextToImage] 任务 ${taskId} 状态: ${status} (第 ${i + 1} 次查询)`,
    );

    if (status === "SUCCEEDED") {
      const imageUrl = data?.output?.results?.[0]?.url;
      if (!imageUrl) {
        throw new Error("任务成功但未返回图片 URL");
      }
      return imageUrl;
    }

    if (status === "FAILED") {
      const errMsg = data?.output?.message || "未知错误";
      throw new Error(`图片生成失败: ${errMsg}`);
    }

    // PENDING 或 RUNNING 继续轮询
  }

  throw new Error("图片生成超时");
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const { prompt } = await req.json();

    if (!prompt || typeof prompt !== "string" || prompt.trim().length === 0) {
      return errorResponse("缺少或无效的 'prompt' 参数", 400);
    }

    if (prompt.length > 500) {
      return errorResponse("描述文本过长（最大 500 字符）", 400);
    }

    const apiKey = Deno.env.get("DASHSCOPE_API_KEY");
    if (!apiKey) {
      return errorResponse("未配置 DASHSCOPE_API_KEY", 500);
    }

    console.log(`[TextToImage] 开始生成图片: "${prompt}"`);

    // 提交任务
    const taskId = await submitImageTask(apiKey, prompt);

    // 轮询结果
    const imageUrl = await pollTaskResult(apiKey, taskId);

    console.log(`[TextToImage] 图片生成成功: ${imageUrl}`);

    return new Response(
      JSON.stringify({
        success: true,
        image_url: imageUrl,
        task_id: taskId,
      }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      },
    );
  } catch (e) {
    return errorResponse(e instanceof Error ? e.message : String(e), 500);
  }
});

console.log("[TextToImage] Edge Function 已初始化完成，等待请求...");
