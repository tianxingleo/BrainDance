import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import {
  matchesAssetSearchQuery,
  normalizeAssetSearchKeywords,
} from "./assetTools.ts";

Deno.test("normalizeAssetSearchKeywords 会去掉主题检索尾词并保留核心中文关键词", () => {
  assertEquals(normalizeAssetSearchKeywords("找一下手办相关的"), ["手办"]);
  assertEquals(normalizeAssetSearchKeywords("请找初音未来相关的模型"), [
    "初音未来",
  ]);
});

Deno.test("matchesAssetSearchQuery 会用核心关键词匹配资产文本", () => {
  assertEquals(
    matchesAssetSearchQuery("找一下手办相关的", {
      scene_id: "scene_demo",
      display_name: "展示柜手办合集",
      description: "书桌旁边的动漫手办模型",
      tags: ["手办", "动漫"],
      objects: ["摆件"],
    }),
    true,
  );
  assertEquals(
    matchesAssetSearchQuery("找一下手办相关的", {
      scene_id: "scene_room",
      display_name: "办公室扫描",
      description: "会议桌和白板",
      tags: ["会议室"],
      objects: ["桌子"],
    }),
    false,
  );
});
