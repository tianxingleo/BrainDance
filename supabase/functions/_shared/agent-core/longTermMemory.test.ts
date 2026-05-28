import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import {
  buildLongTermMemorySignal,
  shouldPersistLongTermMemory,
} from "./longTermMemory.ts";

Deno.test("buildLongTermMemorySignal 会从当前 turn 独立抽取 region/object/time/type 信号", () => {
  const signal = buildLongTermMemorySignal("帮我找客厅里的杯子，看看最近有没有变化", {
    mode: "spatial_search",
    answer: "已找到相关内容",
    intent: null,
    top_candidates: [
      {
        description: "客厅桌上的杯子",
        display_name: "客厅杯子",
        objects: ["杯子"],
        tags: ["客厅", "桌面"],
      },
    ],
    evidence: {
      tags: ["客厅", "桌面"],
    },
  });

  assertEquals(signal.responseMode, "spatial_search");
  assertEquals(signal.topResultSummary, "客厅桌上的杯子");
  assertEquals(signal.regions.includes("客厅"), true);
  assertEquals(signal.objects.includes("杯子"), true);
  assertEquals(signal.timeRanges.includes("最近"), true);
  assertEquals(signal.assetTypes.includes("scene"), true);
});

Deno.test("shouldPersistLongTermMemory 现在只依赖当前 turn 的长期记忆信号", () => {
  const signal = buildLongTermMemorySignal("帮我找模型", {
    mode: "asset_metadata",
    answer: "ok",
    intent: null,
    top_candidates: [],
    evidence: null,
  });

  assertEquals(
    shouldPersistLongTermMemory(
      {
        preferredRegions: [],
        preferredAssetTypes: [],
        preferredTimeRanges: [],
        preferredObjects: [],
        recentSearches: [],
        searchCount: 8,
        lastUpdatedAt: "2026-05-22T00:00:00Z",
      },
      signal,
    ),
    true,
  );
});
