import {
  assertEquals,
  assertExists,
} from "https://deno.land/std@0.168.0/testing/asserts.ts";
import {
  buildCompareActions,
  buildCompareDiff,
  normalizeCompareWindows,
} from "./agent.ts";

Deno.test("normalizeCompareWindows 会在只给目标窗口时补齐基线窗口", () => {
  const result = normalizeCompareWindows({
    searchText: "书桌",
    compareFocus: "房间变化",
    baselineStartTime: null,
    baselineEndTime: null,
    targetStartTime: "2026-03-20T00:00:00Z",
    targetEndTime: "2026-03-25T00:00:00Z",
    reasoning: "用户给出了最近窗口",
  });

  assertEquals(result.target.startTime, "2026-03-20T00:00:00Z");
  assertEquals(result.target.endTime, "2026-03-25T00:00:00Z");
  assertExists(result.baseline.startTime);
  assertEquals(result.baseline.endTime, "2026-03-20T00:00:00Z");
});

Deno.test("buildCompareDiff 会输出新增和移除的对象与标签", () => {
  const diff = buildCompareDiff({
    sceneId: "scene-old",
    modelId: "model-old",
    userId: "u1",
    displayName: "旧场景",
    description: "旧桌面",
    createdAt: "2026-03-10T10:00:00Z",
    similarity: 0.8,
    objects: ["耳机", "键盘"],
    tags: ["桌面", "靠窗"],
    plyPath: null,
    bestFrame: null,
  }, {
    sceneId: "scene-new",
    modelId: "model-new",
    userId: "u1",
    displayName: "新场景",
    description: "新桌面",
    createdAt: "2026-03-25T10:00:00Z",
    similarity: 0.9,
    objects: ["耳机", "台灯"],
    tags: ["桌面", "夜间"],
    plyPath: null,
    bestFrame: null,
  });

  assertEquals(diff.commonObjects, ["耳机"]);
  assertEquals(diff.addedObjects, ["台灯"]);
  assertEquals(diff.removedObjects, ["键盘"]);
  assertEquals(diff.commonTags, ["桌面"]);
  assertEquals(diff.addedTags, ["夜间"]);
  assertEquals(diff.removedTags, ["靠窗"]);
});

Deno.test("buildCompareActions 会为场景生成打开与飞行动作", () => {
  const actions = buildCompareActions({
    baseline: {
      sceneId: "scene-old",
      modelId: "model-old",
      userId: "u1",
      displayName: "旧场景",
      description: "旧桌面",
      createdAt: "2026-03-10T10:00:00Z",
      similarity: 0.8,
      objects: [],
      tags: [],
      plyPath: "u1/scene-old/output/point_cloud.ply",
      bestFrame: {
        imageName: "frame_0001.jpg",
        similarity: 0.8,
        transformMatrix: [1, 0, 0, 1],
        tag: "书桌近景",
      },
    },
    target: null,
    supabase: {
      storage: {
        from() {
          return {
            getPublicUrl(path: string) {
              return { data: { publicUrl: `https://example.com/${path}` } };
            },
          };
        },
      },
    } as never,
    bucket: "braindance-assets",
  });

  assertEquals(actions.length, 2);
  assertEquals(actions[0]?.type, "open_scene");
  assertEquals(actions[1]?.type, "fly_to_pose");
});
