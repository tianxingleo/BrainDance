import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import {
  buildAssetAnswer,
  collectAssetToolResult,
  createEmptyAssetToolState,
} from "./assetTools.ts";

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
