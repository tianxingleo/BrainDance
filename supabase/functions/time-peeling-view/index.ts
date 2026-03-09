import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

function makeError(message: string, status = 400): Response {
  return new Response(JSON.stringify({ success: false, error: message }), {
    status,
    headers: {
      ...corsHeaders,
      "Content-Type": "application/json",
    },
  });
}

function toPublicUrl(supabaseUrl: string, storagePath: string): string {
  if (storagePath.startsWith("http://") || storagePath.startsWith("https://")) {
    return storagePath;
  }
  const base = supabaseUrl.replace(/\/+$/, "");
  return `${base}/storage/v1/object/public/braindance-assets/${storagePath}`;
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const authHeader = req.headers.get("Authorization");
    if (!authHeader?.startsWith("Bearer ")) {
      return makeError("缺少授权信息", 401);
    }
    const jwt = authHeader.replace("Bearer ", "");

    const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

    if (!supabaseUrl || !serviceKey) {
      return makeError("缺少 Supabase 环境变量", 500);
    }

    const admin = createClient(supabaseUrl, serviceKey);
    const { data: userData, error: userError } = await admin.auth.getUser(jwt);
    if (userError || !userData?.user?.id) {
      return makeError("无效登录态", 401);
    }
    const userId = userData.user.id;

    const body = await req.json();
    const {
      space_id: spaceId,
      left_capture_id: leftCaptureId,
      right_capture_id: rightCaptureId,
    } = body ?? {};

    if (!spaceId || !leftCaptureId || !rightCaptureId) {
      return makeError("缺少必要参数: space_id / left_capture_id / right_capture_id");
    }

    const { data: space, error: spaceErr } = await admin
      .from("memory_spaces")
      .select("id, user_id")
      .eq("id", spaceId)
      .single();

    if (spaceErr || !space) {
      return makeError("空间不存在", 404);
    }
    if (space.user_id !== userId) {
      return makeError("无权访问该空间", 403);
    }

    const { data: captures, error: capErr } = await admin
      .from("space_captures")
      .select("id, scene_id, status, alignment_matrix, alignment_score")
      .in("id", [leftCaptureId, rightCaptureId])
      .eq("space_id", spaceId)
      .eq("user_id", userId);

    if (capErr || !captures || captures.length !== 2) {
      return makeError("时间切片不存在或不属于同一空间", 404);
    }

    const captureById = new Map(captures.map((c) => [c.id, c]));
    const leftCapture = captureById.get(leftCaptureId);
    const rightCapture = captureById.get(rightCaptureId);
    if (!leftCapture || !rightCapture) {
      return makeError("切片参数无效", 400);
    }

    const { data: assets, error: assetErr } = await admin
      .from("model_assets")
      .select("capture_id, ply_path, meta_info")
      .in("capture_id", [leftCaptureId, rightCaptureId])
      .eq("user_id", userId);

    if (assetErr || !assets) {
      return makeError("读取模型资产失败", 500);
    }

    const assetByCapture = new Map(assets.map((a) => [a.capture_id, a]));
    const leftAsset = assetByCapture.get(leftCaptureId);
    const rightAsset = assetByCapture.get(rightCaptureId);

    if (!leftAsset?.ply_path || !rightAsset?.ply_path) {
      return makeError("至少一个切片没有可渲染模型", 404);
    }

    let initialPose = null;
    if (leftAsset.meta_info && typeof leftAsset.meta_info === "object") {
      initialPose = (leftAsset.meta_info as Record<string, unknown>).initial_camera_pose ?? null;
    }

    return new Response(
      JSON.stringify({
        success: true,
        space_id: spaceId,
        base_capture_id: leftCaptureId,
        overlay_capture_id: rightCaptureId,
        base_model: toPublicUrl(supabaseUrl, leftAsset.ply_path),
        overlay_model: toPublicUrl(supabaseUrl, rightAsset.ply_path),
        overlay_alignment_matrix: rightCapture.alignment_matrix ?? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        alignment_score: rightCapture.alignment_score,
        default_alpha: 0.5,
        initial_pose: initialPose,
      }),
      {
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
        },
      },
    );
  } catch (e) {
    return makeError(e instanceof Error ? e.message : String(e), 500);
  }
});
