import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import {
  buildAssetAnswer,
  buildComparisonResult,
  createEmptyAssetToolState,
  type ModelAssetBundle,
  renderDisplayNameTemplate,
} from "../_shared/agent-core/assetTools.ts";

Deno.test("renderDisplayNameTemplate 支持 scene_id 与 created_date 占位符", () => {
  const displayName = renderDisplayNameTemplate(
    "宿舍-{{created_date}}-{{scene_id}}",
    {
      scene_id: "scene_a",
      display_name: "旧名字",
      created_at: "2026-03-22T08:00:00Z",
    },
    0,
  );

  assertEquals(displayName, "宿舍-2026-03-22-scene_a");
});

Deno.test("buildComparisonResult 会输出共同标签和时间顺序", () => {
  const rows: ModelAssetBundle[] = [
    {
      id: "m1",
      scene_id: "scene-1",
      display_name: "宿舍-搬家前",
      description: "第一版",
      objects: ["书桌", "台灯", "耳机"],
      tags: ["宿舍", "2025秋"],
      created_at: "2026-03-20T08:00:00Z",
      preview_img_path: null,
      ply_path: null,
      meta_info: {},
      pose_count: 12,
    },
    {
      id: "m2",
      scene_id: "scene-2",
      display_name: "宿舍-搬家后",
      description: "第二版",
      objects: ["书桌", "纸箱"],
      tags: ["宿舍", "2026春"],
      created_at: "2026-03-22T08:00:00Z",
      preview_img_path: null,
      ply_path: null,
      meta_info: {},
      pose_count: 18,
    },
  ];

  const result = buildComparisonResult(rows);

  assertEquals(result.diff.common_tags, ["宿舍"]);
  assertEquals(result.diff.common_objects, ["书桌"]);
  assertEquals(result.diff.time_order, ["m1", "m2"]);
  assertEquals(result.diff.object_only_by_model.m2, ["纸箱"]);
});

Deno.test("buildAssetAnswer 在 bundle 场景下返回可读的模型概览", () => {
  const state = createEmptyAssetToolState();
  state.bundle = [
    {
      id: "m1",
      scene_id: "scene-1",
      display_name: "宿舍书桌",
      description: "桌面比较整洁，书和台灯都在",
      objects: ["书桌", "台灯"],
      tags: ["宿舍", "书桌"],
      created_at: "2026-03-20T08:00:00Z",
      preview_img_path: null,
      ply_path: null,
      meta_info: {},
      pose_count: 12,
    },
    {
      id: "m2",
      scene_id: "scene-2",
      display_name: "客厅沙发",
      description: "适合做客厅陈设回顾",
      objects: ["沙发", "茶几"],
      tags: ["客厅"],
      created_at: "2026-03-22T08:00:00Z",
      preview_img_path: null,
      ply_path: null,
      meta_info: {},
      pose_count: 8,
    },
  ];

  const answer = buildAssetAnswer(state);

  assertEquals(
    answer,
    "我先整理出 2 个可参考的模型：\n" +
      "1. 宿舍书桌：桌面比较整洁，书和台灯都在；标签：宿舍、书桌；pose 12 个\n" +
      "2. 客厅沙发：适合做客厅陈设回顾；标签：客厅；pose 8 个\n" +
      "如果你想继续缩小范围，我可以再按时间、标签或场景帮你筛一轮。",
  );
});

Deno.test("buildAssetAnswer 在推荐问句下只返回前五个推荐模型", () => {
  const state = createEmptyAssetToolState();
  state.list = Array.from({ length: 6 }, (_, index) => ({
    id: `m${index + 1}`,
    scene_id: `scene-${index + 1}`,
    display_name: `模型-${index + 1}`,
    description: null,
    tags: index < 5 ? ["标签A", `标签${index + 1}`] : [],
    created_at: `2026-03-${String(20 + index).padStart(2, "0")}T08:00:00Z`,
  }));

  const answer = buildAssetAnswer(state, {
    query: "有什么推荐的模型？",
  });

  assertEquals(
    answer,
    "我先从当前候选里推荐这 5 个模型：\n" +
      "1. 模型-5：标签较完整（标签A、标签5）\n" +
      "2. 模型-4：标签较完整（标签A、标签4）\n" +
      "3. 模型-3：标签较完整（标签A、标签3）\n" +
      "4. 模型-2：标签较完整（标签A、标签2）\n" +
      "5. 模型-1：标签较完整（标签A、标签1）\n" +
      "如果你需要更精确的推荐，我可以继续读取其中几个模型的详细摘要再细分。",
  );
});
