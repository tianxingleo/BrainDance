import {
  assertEquals,
  assertExists,
} from "https://deno.land/std@0.168.0/testing/asserts.ts";
import {
  buildGeneralAssistantFallbackAnswer,
  buildResponseResolutionFromResponse,
  buildVisualizationActions,
  isAssetDiscoveryQuery,
  isDirectReplyQuery,
  normalizeExplicitTimeRange,
  parseModelPresentation,
  parseSpatialIntentHeuristically,
  parseDeterministicAssetRenameIntent,
  pickSpatialSearchAnswerAfterStop,
  scoreSceneCandidate,
  shouldStopAssetToolLoop,
  shouldForceAnotherToolRound,
  shouldPreferHeuristicSpatialRoute,
  summarizeCandidateEvidence,
} from "../_shared/agent-core/spatialAgent.ts";

Deno.test("isDirectReplyQuery 会识别纯问候与致谢", () => {
  assertEquals(isDirectReplyQuery("你好"), true);
  assertEquals(isDirectReplyQuery("谢谢！"), true);
  assertEquals(isDirectReplyQuery("你是谁"), false);
  assertEquals(isDirectReplyQuery("你好，帮我找一下红色杯子"), false);
});

Deno.test("buildGeneralAssistantFallbackAnswer 会在无候选时输出通用 Agent 回答", async () => {
  const fakeModel = {
    async invoke() {
      return {
        content:
          "我是 BrainDance 的空间记忆智能管理助手，可以帮你检索场景、比较时间变化，也能整理模型资产。",
      };
    },
  };

  assertEquals(
    (
      await buildGeneralAssistantFallbackAnswer(
        fakeModel,
        "你是谁",
      )
    ).includes("空间记忆智能管理助手"),
    true,
  );
});

Deno.test("buildResponseResolutionFromResponse 会为通用 fallback 标注 general_fallback", () => {
  const resolution = buildResponseResolutionFromResponse({
    success: true,
    mode: "spatial_search",
    intent: {
      rewrittenQuery: "你是谁",
      targetType: "scene",
      objectHint: null,
      locationHint: null,
      sceneHint: null,
      timeHint: null,
      startTime: null,
      endTime: null,
      reasoning: "测试 fallback",
    },
    selection: {
      scene_id: null,
      model_id: null,
      pose_image_id: null,
      confidence: 0,
      reason: "当前没有可信检索候选，已回退为通用 Agent 自然语言回答。",
    },
    answer: "我是 BrainDance 的空间记忆智能管理助手。",
    actions: [],
    viewer_payload: {
      ply: null,
      poses: null,
      matrix: null,
      imageId: null,
    },
    evidence: null,
    candidates: [],
    top_candidates: [],
    selected_candidate_reason: "未命中可信候选，已回退为通用 Agent 回答。",
    tool_trace: [],
    asset_context: {
      last_tool_name: null,
      list: null,
      bundle: null,
      comparison: null,
      operation: null,
      pose_summary: null,
      related_models: null,
      place_versions: null,
      collection_summary: null,
    },
    compare_context: null,
    collection_context: null,
    creative_context: null,
    memory_graph_context: null,
  });

  assertEquals(resolution.kind, "general_fallback");
});

Deno.test("shouldStopAssetToolLoop 会在生成写入预览后停止", () => {
  const decision = shouldStopAssetToolLoop({
    state: {
      lastToolName: "batch_patch_model_metadata",
      list: null,
      bundle: null,
      comparison: null,
      duplicateNames: null,
      operation: {
        tool_name: "batch_patch_model_metadata",
        dry_run: true,
        requires_confirmation: true,
        affected_count: 2,
        preview: [],
      },
      poseSummary: null,
      relatedModels: null,
      placeVersions: null,
      collectionSummary: null,
    },
    trace: [{
      toolName: "batch_patch_model_metadata",
      args: {},
      resultSummary: "已生成预览",
    }],
  });

  assertEquals(decision.stop, true);
});

Deno.test("shouldStopAssetToolLoop 会在多轮只剩列表读取时停止", () => {
  const decision = shouldStopAssetToolLoop({
    state: {
      lastToolName: "read_model_assets",
      list: [{
        id: "550e8400-e29b-41d4-a716-446655440000",
        scene_id: "scene-1",
        display_name: "宿舍书桌",
        description: null,
        tags: [],
        created_at: "2026-03-27T10:00:00Z",
      }],
      bundle: null,
      comparison: null,
      duplicateNames: null,
      operation: null,
      poseSummary: null,
      relatedModels: null,
      placeVersions: null,
      collectionSummary: null,
    },
    trace: [
      {
        toolName: "read_model_assets",
        args: { query: "宿舍" },
        resultSummary: "读取到 1 个模型资产",
      },
      {
        toolName: "read_model_assets",
        args: { query: "书桌" },
        resultSummary: "读取到 1 个模型资产",
      },
    ],
  });

  assertEquals(decision.stop, true);
});

Deno.test("shouldPreferHeuristicSpatialRoute 会让简单查找语句避开 LLM 路由", () => {
  assertEquals(shouldPreferHeuristicSpatialRoute("查一下电脑"), false);
  assertEquals(shouldPreferHeuristicSpatialRoute("看一下桌上的杯子"), true);
  assertEquals(shouldPreferHeuristicSpatialRoute("找上周拍的红色杯子"), true);
  assertEquals(shouldPreferHeuristicSpatialRoute("找一个会议室资产"), false);
  assertEquals(shouldPreferHeuristicSpatialRoute("找初音未来相关的"), false);
  assertEquals(
    shouldPreferHeuristicSpatialRoute("比较上周和现在的客厅变化"),
    false,
  );
  assertEquals(shouldPreferHeuristicSpatialRoute("谢谢"), false);
});

Deno.test("isAssetDiscoveryQuery 会识别模型资产级查找请求", () => {
  assertEquals(isAssetDiscoveryQuery("找一个会议室资产"), true);
  assertEquals(isAssetDiscoveryQuery("帮我找个办公室模型"), true);
  assertEquals(isAssetDiscoveryQuery("找初音未来相关的"), true);
  assertEquals(isAssetDiscoveryQuery("会议室里的投影仪在哪"), false);
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

Deno.test("parseSpatialIntentHeuristically 虽可识别会议室资产为场景目标，但不应直接触发空间快路径", () => {
  const intent = parseSpatialIntentHeuristically("找一个会议室资产");

  assertEquals(intent.targetType, "scene");
  assertEquals(intent.rewrittenQuery, "找一个会议室资产");
  assertEquals(intent.sceneHint, "找一个会议室资产");
});

Deno.test("parseSpatialIntentHeuristically 会处理相对时间并生成时间范围", () => {
  const intent = parseSpatialIntentHeuristically("找最近拍的会议室");

  assertEquals(intent.targetType, "time");
  assertExists(intent.startTime);
  assertExists(intent.endTime);
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

Deno.test("pickSpatialSearchAnswerAfterStop 优先使用 stop_search 后的用户可读总结", () => {
  const answer = pickSpatialSearchAnswerAfterStop({
    stopSummary: "我已经找到最相关的桌面模型，可以直接打开查看。",
  });

  assertEquals(answer, "我已经找到最相关的桌面模型，可以直接打开查看。");
});

Deno.test("pickSpatialSearchAnswerAfterStop 在没有 stop_search 总结时给出中性兜底", () => {
  const answer = pickSpatialSearchAnswerAfterStop({
    stopSummary: "",
  });

  assertEquals(answer, "已为你整理了相关空间候选，请在结果区继续查看。");
});

Deno.test("shouldForceAnotherToolRound 在单个高分交叉证据候选时不再强制续轮", () => {
  const candidates = new Map([
    ["scene-strong", {
      modelId: "m-strong",
      sceneId: "scene-strong",
      userId: "u1",
      description: "宿舍书桌上的黑色耳机",
      objects: ["耳机"],
      tags: ["书桌", "黑色耳机"],
      plyPath: "u1/scene-strong/output/point_cloud.ply",
      previewImgPath: null,
      createdAt: "2026-03-25T08:00:00Z",
      metaInfo: {},
      sourceScores: {
        pose_semantic_search: 0.91,
        scene_metadata_search: 0.79,
      },
      bestPose: {
        image_name: "frame_0010.jpg",
        transform_matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        similarity: 0.91,
        tag: "黑色耳机近景",
      },
    }],
  ]);

  const shouldContinue = shouldForceAnotherToolRound({
    intent: {
      rewrittenQuery: "黑色耳机在哪",
      targetType: "object",
      objectHint: "黑色耳机",
      locationHint: null,
      sceneHint: null,
      timeHint: null,
      startTime: null,
      endTime: null,
      reasoning: "已经拿到高分候选，只需要进入最终裁决。",
    },
    candidates,
    trace: [
      {
        toolName: "pose_semantic_search",
        args: { query: "黑色耳机在哪" },
        resultSummary: "pose_semantic_search 返回 1 条候选",
      },
      {
        toolName: "scene_metadata_search",
        args: { query: "黑色耳机在哪" },
        resultSummary: "scene_metadata_search 返回 1 条候选",
      },
    ],
  });

  assertEquals(shouldContinue, false);
});

Deno.test("parseModelPresentation 在 spatial_search 模式下使用更小默认值", () => {
  const presentation = parseModelPresentation("找一下红色杯子", {
    mode: "spatial_search",
  });
  assertEquals(presentation.requested_model_count, null);
  assertEquals(presentation.effective_model_count, 3);
  assertEquals(presentation.default_model_count, 3);
  assertEquals(presentation.max_model_count, 10);
  assertEquals(presentation.source, "default");
});

Deno.test("parseModelPresentation 在 spatial_search 模式下也会 clamp 到 max 10", () => {
  const presentation = parseModelPresentation("给我看 30 个候选场景", {
    mode: "spatial_search",
  });
  assertEquals(presentation.requested_model_count, 30);
  assertEquals(presentation.effective_model_count, 10);
  assertEquals(presentation.source, "clamped");
});
