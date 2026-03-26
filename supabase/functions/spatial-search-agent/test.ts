import {
  assertEquals,
  assertExists,
} from "https://deno.land/std@0.168.0/testing/asserts.ts";
import {
  buildVisualizationActions,
  isDirectReplyQuery,
  normalizeExplicitTimeRange,
  parseDeterministicAssetRenameIntent,
  scoreSceneCandidate,
  shouldForceAnotherToolRound,
  shouldPreferHeuristicSpatialRoute,
  summarizeCandidateEvidence,
} from "../_shared/agent-core/spatialAgent.ts";

Deno.test("isDirectReplyQuery 会识别纯问候与致谢", () => {
  assertEquals(isDirectReplyQuery("你好"), true);
  assertEquals(isDirectReplyQuery("谢谢！"), true);
  assertEquals(isDirectReplyQuery("你好，帮我找一下红色杯子"), false);
});

Deno.test("shouldPreferHeuristicSpatialRoute 会让简单查找语句避开 LLM 路由", () => {
  assertEquals(shouldPreferHeuristicSpatialRoute("查一下电脑"), true);
  assertEquals(shouldPreferHeuristicSpatialRoute("看一下桌上的杯子"), true);
  assertEquals(
    shouldPreferHeuristicSpatialRoute("比较上周和现在的客厅变化"),
    false,
  );
  assertEquals(shouldPreferHeuristicSpatialRoute("谢谢"), false);
});

Deno.test("parseDeterministicAssetRenameIntent 会识别最新模型改名但缺少新名字", () => {
  const intent = parseDeterministicAssetRenameIntent("我想修改最新的模型名字");

  assertExists(intent);
  assertEquals(intent?.target.kind, "latest");
  assertEquals(intent?.newName, null);
});

Deno.test("parseDeterministicAssetRenameIntent 会提取最新模型的新名字", () => {
  const intent = parseDeterministicAssetRenameIntent(
    "把最新的模型改名为宿舍-午后版本",
  );

  assertExists(intent);
  assertEquals(intent?.target.kind, "latest");
  if (intent?.target.kind === "latest") {
    assertEquals(intent.target.count, 1);
  }
  assertEquals(intent?.newName, "宿舍-午后版本");
});

Deno.test("parseDeterministicAssetRenameIntent 会提取最新三个模型的批量改名意图", () => {
  const intent = parseDeterministicAssetRenameIntent(
    "把最新三个模型改名为宿舍-归档版",
  );

  assertExists(intent);
  assertEquals(intent?.target.kind, "latest");
  if (intent?.target.kind === "latest") {
    assertEquals(intent.target.count, 3);
  }
  assertEquals(intent?.newName, "宿舍-归档版");
});

Deno.test("parseDeterministicAssetRenameIntent 会在单选模型时走定向改名", () => {
  const intent = parseDeterministicAssetRenameIntent("把它重命名为书桌近景", {
    selectedModelIds: ["550e8400-e29b-41d4-a716-446655440000"],
  });

  assertExists(intent);
  assertEquals(intent?.target.kind, "selected");
  if (intent?.target.kind === "selected") {
    assertEquals(intent.target.modelIds.length, 1);
  }
  assertEquals(intent?.newName, "书桌近景");
});

Deno.test("parseDeterministicAssetRenameIntent 会在多选模型时走批量改名", () => {
  const intent = parseDeterministicAssetRenameIntent(
    "把这几个模型改名为宿舍批次",
    {
      selectedModelIds: [
        "550e8400-e29b-41d4-a716-446655440000",
        "550e8400-e29b-41d4-a716-446655440001",
      ],
    },
  );

  assertExists(intent);
  assertEquals(intent?.target.kind, "selected");
  if (intent?.target.kind === "selected") {
    assertEquals(intent.target.modelIds.length, 2);
  }
  assertEquals(intent?.newName, "宿舍批次");
});

Deno.test("parseDeterministicAssetRenameIntent 会利用上一轮会话中的单模型上下文", () => {
  const intent = parseDeterministicAssetRenameIntent(
    "把它改名为宿舍书桌-最终版",
    {
      sessionState: {
        lastMode: "asset_metadata",
        lastSelectedModelIds: ["550e8400-e29b-41d4-a716-446655440000"],
      },
    },
  );

  assertExists(intent);
  assertEquals(intent?.target.kind, "session");
  if (intent?.target.kind === "session") {
    assertEquals(intent.target.modelIds.length, 1);
  }
  assertEquals(intent?.newName, "宿舍书桌-最终版");
});

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

Deno.test("buildVisualizationActions 会生成打开场景与飞行动作", () => {
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

  assertEquals(actions.length, 2);
  assertEquals(actions[0]?.type, "open_scene");
  assertEquals(actions[1]?.type, "fly_to_pose");
});

Deno.test("summarizeCandidateEvidence 会识别多来源证据与最高分", () => {
  const candidates = new Map([
    ["scene-demo", {
      modelId: "m1",
      sceneId: "scene-demo",
      userId: "u1",
      description: "桌面上的红色杯子",
      objects: ["杯子"],
      tags: ["桌面"],
      plyPath: "u1/scene-demo/output/point_cloud.ply",
      previewImgPath: null,
      createdAt: "2026-03-24T08:00:00Z",
      metaInfo: {},
      sourceScores: {
        pose_semantic_search: 0.88,
        scene_metadata_search: 0.74,
      },
      bestPose: {
        image_name: "frame_0001.jpg",
        transform_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        similarity: 0.88,
        tag: "红色杯子近景",
      },
    }],
  ]);

  const evidence = summarizeCandidateEvidence(candidates, {
    rewrittenQuery: "红色杯子",
    targetType: "object",
  });

  assertEquals(evidence.candidateCount, 1);
  assertEquals(evidence.hasMultiSourceEvidence, true);
  assertEquals(evidence.topScore > 0.62, true);
});

Deno.test("shouldForceAnotherToolRound 在单来源低覆盖时返回 true", () => {
  const candidates = new Map([
    ["scene-demo", {
      modelId: "m1",
      sceneId: "scene-demo",
      userId: "u1",
      description: "桌面上的红色杯子",
      objects: [],
      tags: ["红色杯子近景"],
      plyPath: "u1/scene-demo/output/point_cloud.ply",
      previewImgPath: null,
      createdAt: "2026-03-24T08:00:00Z",
      metaInfo: {},
      sourceScores: {
        pose_semantic_search: 0.55,
      },
      bestPose: {
        image_name: "frame_0001.jpg",
        transform_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        similarity: 0.55,
        tag: "红色杯子近景",
      },
    }],
  ]);

  const shouldContinue = shouldForceAnotherToolRound({
    intent: {
      rewrittenQuery: "红色杯子",
      targetType: "object",
      objectHint: "杯子",
      locationHint: null,
      sceneHint: null,
      timeHint: null,
      startTime: null,
      endTime: null,
      reasoning: "先找物体，再补场景证据。",
    },
    candidates,
    trace: [{
      toolName: "pose_semantic_search",
      args: { query: "红色杯子" },
      resultSummary: "pose_semantic_search 返回 1 条候选",
    }],
  });

  assertEquals(shouldContinue, true);
});

Deno.test("shouldForceAnotherToolRound 在证据充分时返回 false", () => {
  const candidates = new Map([
    ["scene-1", {
      modelId: "m1",
      sceneId: "scene-1",
      userId: "u1",
      description: "桌面上的红色杯子",
      objects: ["杯子"],
      tags: ["桌面"],
      plyPath: "u1/scene-1/output/point_cloud.ply",
      previewImgPath: null,
      createdAt: "2026-03-24T08:00:00Z",
      metaInfo: {},
      sourceScores: {
        pose_semantic_search: 0.92,
        scene_metadata_search: 0.82,
      },
      bestPose: {
        image_name: "frame_0001.jpg",
        transform_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        similarity: 0.92,
        tag: "红色杯子近景",
      },
    }],
    ["scene-2", {
      modelId: "m2",
      sceneId: "scene-2",
      userId: "u1",
      description: "桌角附近的水杯",
      objects: ["水杯"],
      tags: ["桌角"],
      plyPath: "u1/scene-2/output/point_cloud.ply",
      previewImgPath: null,
      createdAt: "2026-03-24T07:00:00Z",
      metaInfo: {},
      sourceScores: {
        pose_semantic_search: 0.8,
        scene_metadata_search: 0.76,
      },
      bestPose: {
        image_name: "frame_0002.jpg",
        transform_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        similarity: 0.8,
        tag: "桌角水杯",
      },
    }],
    ["scene-3", {
      modelId: "m3",
      sceneId: "scene-3",
      userId: "u1",
      description: "柜子边上的咖啡杯",
      objects: ["咖啡杯"],
      tags: ["柜子"],
      plyPath: "u1/scene-3/output/point_cloud.ply",
      previewImgPath: null,
      createdAt: "2026-03-24T06:00:00Z",
      metaInfo: {},
      sourceScores: {
        pose_semantic_search: 0.78,
        scene_metadata_search: 0.72,
      },
      bestPose: {
        image_name: "frame_0003.jpg",
        transform_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        similarity: 0.78,
        tag: "柜子边咖啡杯",
      },
    }],
  ]);

  const shouldContinue = shouldForceAnotherToolRound({
    intent: {
      rewrittenQuery: "红色杯子",
      targetType: "object",
      objectHint: "杯子",
      locationHint: null,
      sceneHint: null,
      timeHint: null,
      startTime: null,
      endTime: null,
      reasoning: "已经拿到足够多的候选和多来源证据。",
    },
    candidates,
    trace: [
      {
        toolName: "pose_semantic_search",
        args: { query: "红色杯子" },
        resultSummary: "pose_semantic_search 返回 3 条候选",
      },
      {
        toolName: "scene_metadata_search",
        args: { query: "红色杯子" },
        resultSummary: "scene_metadata_search 返回 3 条候选",
      },
    ],
  });

  assertEquals(shouldContinue, false);
});
