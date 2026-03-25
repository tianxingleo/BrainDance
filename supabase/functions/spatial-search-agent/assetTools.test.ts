import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import {
  buildComparisonResult,
  type ModelAssetBundle,
  renderDisplayNameTemplate,
} from "./assetTools.ts";

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
