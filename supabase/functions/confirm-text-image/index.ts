/**
 * BrainDance 确认文生图片 Edge Function
 *
 * 功能:
 * 1. 下载 DashScope 生成的临时图片
 * 2. 调用 Qwen VL 多模态模型分类：物体 or 场景
 * 3. 上传图片到 Supabase Storage
 * 4. 创建 processing_tasks 记录
 *
 * 测试:
 *   curl -X POST http://127.0.0.1:54321/functions/v1/confirm-text-image \
 *     -H 'Content-Type: application/json' \
 *     -H 'Authorization: Bearer <user_token>' \
 *     -d '{"image_url":"https://...","prompt":"一个红色杯子"}'
 */

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

const DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1";

function errorResponse(message: string, status = 500): Response {
  console.error(`[ConfirmTextImage] 错误: ${message}`);
  return new Response(
    JSON.stringify({ success: false, error: message }),
    {
      status,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    },
  );
}

/**
 * 生成场景 ID，格式与 Flutter 端一致
 */
function generateSceneId(): string {
  const now = new Date();
  const pad = (n: number, len: number) => String(n).padStart(len, "0");
  const rand = Math.floor(Math.random() * 1000000);
  return `scene_${pad(now.getFullYear(), 4)}${pad(now.getMonth() + 1, 2)}${
    pad(now.getDate(), 2)
  }_${pad(rand, 6)}`;
}

/**
 * 调用 Qwen VL 多模态模型对图片进行物体/场景分类
 */
async function classifyImage(
  apiKey: string,
  imageUrl: string,
): Promise<string> {
  const resp = await fetch(`${DASHSCOPE_API_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "qwen-vl-max",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: { url: imageUrl },
            },
            {
              type: "text",
              text:
                "这张图片的主体是一个独立的物体还是一个场景/环境？请只回答'物体'或'场景'。",
            },
          ],
        },
      ],
    }),
  });

  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`Qwen VL API 错误: ${resp.status} - ${err}`);
  }

  const data = await resp.json();
  const answer = data?.choices?.[0]?.message?.content?.trim() ?? "";

  console.log(`[ConfirmTextImage] VL 分类结果: "${answer}"`);

  // 解析分类结果
  if (answer.includes("场景")) {
    return "single_image_sharp";
  }
  // 默认按物体处理
  return "single_image_sam3d";
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const { image_url, prompt } = await req.json();

    if (!image_url || typeof image_url !== "string") {
      return errorResponse("缺少或无效的 'image_url' 参数", 400);
    }

    // 从 Authorization header 获取用户 token
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) {
      return errorResponse("未提供 Authorization header", 401);
    }

    const apiKey = Deno.env.get("DASHSCOPE_API_KEY");
    const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
    const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

    if (!apiKey) {
      return errorResponse("未配置 DASHSCOPE_API_KEY", 500);
    }

    // 使用用户 token 创建客户端以获取用户信息
    const userToken = authHeader.replace("Bearer ", "");
    const supabaseUser = createClient(supabaseUrl, supabaseServiceKey, {
      global: { headers: { Authorization: `Bearer ${userToken}` } },
    });

    // 验证用户
    const { data: { user }, error: authError } =
      await supabaseUser.auth.getUser(userToken);

    if (authError || !user) {
      return errorResponse("用户认证失败", 401);
    }

    console.log(`[ConfirmTextImage] 用户 ${user.id} 确认图片`);

    // 使用 service role 客户端进行存储和数据库操作
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Step 1: 下载生成的图片
    console.log(`[ConfirmTextImage] 下载图片: ${image_url}`);
    const imageResp = await fetch(image_url);
    if (!imageResp.ok) {
      return errorResponse(`下载图片失败: ${imageResp.status}`, 500);
    }
    const imageBuffer = new Uint8Array(await imageResp.arrayBuffer());

    // Step 2: 分类图片（物体 or 场景）
    console.log("[ConfirmTextImage] 开始分类图片...");
    const taskType = await classifyImage(apiKey, image_url);
    console.log(`[ConfirmTextImage] 分类结果: ${taskType}`);

    // Step 3: 上传图片到 Storage
    const sceneId = generateSceneId();
    const storagePath = `${user.id}/${sceneId}/raw/image.png`;

    console.log(`[ConfirmTextImage] 上传到 Storage: ${storagePath}`);
    const { error: uploadError } = await supabase.storage
      .from("braindance-assets")
      .upload(storagePath, imageBuffer, {
        contentType: "image/png",
        upsert: false,
      });

    if (uploadError) {
      throw new Error(`上传图片失败: ${uploadError.message}`);
    }

    // Step 4: 创建 processing_tasks 记录
    console.log(
      `[ConfirmTextImage] 创建任务: scene_id=${sceneId}, task_type=${taskType}`,
    );
    const { error: insertError } = await supabase
      .from("processing_tasks")
      .insert({
        scene_id: sceneId,
        user_id: user.id,
        status: "pending",
        task_type: taskType,
      });

    if (insertError) {
      throw new Error(`创建任务失败: ${insertError.message}`);
    }

    console.log(`[ConfirmTextImage] 完成! scene_id=${sceneId}`);

    return new Response(
      JSON.stringify({
        success: true,
        scene_id: sceneId,
        task_type: taskType,
      }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      },
    );
  } catch (e) {
    return errorResponse(e instanceof Error ? e.message : String(e), 500);
  }
});

console.log("[ConfirmTextImage] Edge Function 已初始化完成，等待请求...");
