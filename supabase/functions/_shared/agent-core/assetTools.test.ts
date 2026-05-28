import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import {
  buildAssetAnswer,
  collectAssetToolResult,
  createEmptyAssetToolState,
} from "./assetTools.ts";
import { parseModelPresentation } from "./spatialAgent.ts";

Deno.test("collectAssetToolResult 会记录 write_model_assets 的预览结果", () => {
  const state = createEmptyAssetToolState();
  const count = collectAssetToolResult(
    "write_model_assets",
    JSON.stringify({
      kind: "asset_operation",
      operation: {
        tool_name: "write_model_assets",
        dry_run: true,
        requires_confirmation: true,
        affected_count: 2,
        preview: [
          {
            model_id: "550e8400-e29b-41d4-a716-446655440000",
            scene_id: "scene_a",
            old_display_name: "旧名字1",
            new_display_name: "test1",
            old_summary_title: null,
            new_summary_title: "宿舍桌面记忆",
            old_description: null,
            new_description: null,
            old_tags: ["原标签"],
            new_tags: ["原标签"],
          },
          {
            model_id: "550e8400-e29b-41d4-a716-446655440001",
            scene_id: "scene_b",
            old_display_name: "旧名字2",
            new_display_name: "test2",
            old_description: null,
            new_description: null,
            old_tags: [],
            new_tags: [],
          },
        ],
      },
    }),
    state,
  );

  assertEquals(count, 2);
  assertEquals(state.operation?.tool_name, "write_model_assets");
  assertEquals(state.operation?.preview[0]?.new_display_name, "test1");
  assertEquals(state.operation?.preview[0]?.new_summary_title, "宿舍桌面记忆");
  assertEquals(state.operation?.preview[1]?.new_display_name, "test2");
});

Deno.test("buildAssetAnswer 会输出重名模型摘要", () => {
  const answer = buildAssetAnswer({
    ...createEmptyAssetToolState(),
    duplicateNames: [
      {
        display_name: "客厅扫描",
        count: 2,
        rows: [
          {
            id: "550e8400-e29b-41d4-a716-446655440000",
            scene_id: "scene_1",
            display_name: "客厅扫描",
            summary_title: null,
            description: null,
            tags: [],
            created_at: "2026-03-27T10:00:00Z",
          },
          {
            id: "550e8400-e29b-41d4-a716-446655440001",
            scene_id: "scene_2",
            display_name: "客厅扫描",
            summary_title: null,
            description: null,
            tags: [],
            created_at: "2026-03-27T11:00:00Z",
          },
        ],
      },
    ],
  });

  assertEquals(
    answer,
    "当前发现 1 组重名模型：\n1. 客厅扫描：重复 2 次（scene: scene_1、scene_2）\n如果你需要，我可以继续展开这些重名模型的详细摘要。",
  );
});

Deno.test("parseModelPresentation 会从用户语句中识别明确数量", () => {
  const presentation = parseModelPresentation("给我看十个模型", {
    mode: "asset_metadata",
  });
  assertEquals(presentation.requested_model_count, 10);
  assertEquals(presentation.effective_model_count, 10);
  assertEquals(presentation.source, "user_explicit");
  assertEquals(presentation.default_model_count, 5);
  assertEquals(presentation.max_model_count, 20);
});

Deno.test("parseModelPresentation 在超过上限时会 clamp 到 max", () => {
  const presentation = parseModelPresentation("我想看 50 个模型", {
    mode: "asset_metadata",
  });
  assertEquals(presentation.requested_model_count, 50);
  assertEquals(presentation.effective_model_count, 20);
  assertEquals(presentation.source, "clamped");
});

Deno.test("parseModelPresentation 默认情况下使用模式默认值", () => {
  const asset = parseModelPresentation("帮我整理模型", {
    mode: "asset_metadata",
  });
  assertEquals(asset.requested_model_count, null);
  assertEquals(asset.effective_model_count, 5);
  assertEquals(asset.source, "default");

  const spatial = parseModelPresentation("找一下红色杯子", {
    mode: "spatial_search",
  });
  assertEquals(spatial.requested_model_count, null);
  assertEquals(spatial.effective_model_count, 3);
  assertEquals(spatial.default_model_count, 3);
  assertEquals(spatial.max_model_count, 10);
});

Deno.test("buildAssetAnswer 在 displayCount 指定时按数量裁剪 bundle", () => {
  const state = createEmptyAssetToolState();
  state.bundle = Array.from({ length: 6 }, (_, index) => ({
    id: `m${index + 1}`,
    scene_id: `scene-${index + 1}`,
    display_name: `模型-${index + 1}`,
    description: null,
    objects: [],
    tags: ["标签A"],
    created_at: `2026-03-${String(20 + index).padStart(2, "0")}T08:00:00Z`,
    preview_img_path: null,
    ply_path: null,
    meta_info: {},
    pose_count: 0,
  }));

  const answer = buildAssetAnswer(state, { displayCount: 3 });
  assertEquals(typeof answer, "string");
  assertEquals(
    answer!.startsWith("我先整理出 3 个可参考的模型："),
    true,
  );
  assertEquals(answer!.includes("模型-3"), true);
  assertEquals(answer!.includes("模型-4"), false);
  assertEquals(answer!.includes("（共 6 个，已展示前 3 个）"), true);
});
