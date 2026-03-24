import { assertEquals, assertExists } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import {
  buildVisualizationActions,
  normalizeExplicitTimeRange,
  scoreSceneCandidate,
} from "./agent.ts";

Deno.test("normalizeExplicitTimeRange 会处理最近时间语义", () => {
  const result = normalizeExplicitTimeRange({
    timeHint: "最近拍的",
    startTime: null,
    endTime: null,
  });

  assertExists(result.startTime);
  assertExists(result.endTime);
  assertEquals(result.startTime! < result.endTime!, true);
});

Deno.test("scoreSceneCandidate 在物体检索下优先 pose 分数", () => {
  const poseHeavy = scoreSceneCandidate({
    modelId: "m1",
    sceneId: "scene-object",
    userId: "u1",
    description: "桌面上的红色杯子",
    objects: ["杯子"],
    tags: ["桌面"],
    plyPath: "u1/scene-object/output/point_cloud.ply",
    previewImgPath: null,
    createdAt: "2026-03-24T08:00:00Z",
    metaInfo: {},
    sourceScores: {
      pose_semantic_search: 0.92,
      scene_metadata_search: 0.3,
    },
    bestPose: {
      image_name: "frame_0001.jpg",
      transform_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      similarity: 0.92,
      tag: "红色杯子近景",
    },
  }, {
    rewrittenQuery: "红色杯子",
    targetType: "object",
  });

  const sceneHeavy = scoreSceneCandidate({
    modelId: "m2",
    sceneId: "scene-room",
    userId: "u1",
    description: "客厅全景",
    objects: ["沙发"],
    tags: ["客厅"],
    plyPath: "u1/scene-room/output/point_cloud.ply",
    previewImgPath: null,
    createdAt: "2026-03-24T08:00:00Z",
    metaInfo: {},
    sourceScores: {
      pose_semantic_search: 0.25,
      scene_metadata_search: 0.88,
    },
    bestPose: null,
  }, {
    rewrittenQuery: "红色杯子",
    targetType: "object",
  });

  assertEquals(poseHeavy > sceneHeavy, true);
});

Deno.test("buildVisualizationActions 会生成打开模型与飞行动作", () => {
  const actions = buildVisualizationActions({
    scene: {
      modelId: "m1",
      sceneId: "scene-demo",
      userId: "u1",
      description: "演示场景",
      objects: [],
      tags: [],
      plyPath: "u1/scene-demo/output/point_cloud.ply",
      previewImgPath: null,
      createdAt: "2026-03-24T08:00:00Z",
      metaInfo: {},
      sourceScores: {},
      bestPose: null,
    },
    selectedPose: {
      image_name: "frame_0008.jpg",
      transform_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      similarity: 0.88,
      tag: "桌面俯视",
    },
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
    } as unknown as Parameters<typeof buildVisualizationActions>[0]["supabase"],
    bucket: "braindance-assets",
  });

  assertEquals(actions.length, 3);
  assertEquals(actions[0]?.type, "open_model");
  assertEquals(actions[1]?.type, "fly_to_pose");
  assertEquals(actions[2]?.type, "highlight_hotspot");
});
