import {
  assertEquals,
  assertExists,
} from "https://deno.land/std@0.168.0/testing/asserts.ts";
import { runSpatialSearchAgent } from "../_shared/agent-core/spatialAgent.ts";
import { runRecallAgent } from "./agent/recallAgent.ts";
import { buildRecallActionsFromSearchResult } from "./tools/buildViewAction.ts";
import { buildEvidenceFromSpatialResult } from "./tools/getSceneAsset.ts";

Deno.test("buildRecallActionsFromSearchResult 会从搜索结果构造稳定动作协议", () => {
  const actions = buildRecallActionsFromSearchResult({
    success: true,
    intent: {
      original_query: "黑色耳机在哪",
      parsed_search_text: "黑色耳机",
      filter_start: null,
      filter_end: null,
    },
    threshold: 0.5,
    results: [{
      id: "model-a",
      scene_id: "scene-a",
      ply_path: "u1/scene-a/output/point_cloud.ply",
      description: "书桌上的黑色耳机",
      similarity: 0.91,
      matched_frames: [{
        image_name: "frame_01.jpg",
        transform_matrix: [1, 0, 0, 0],
        similarity: 0.91,
        tag: "黑色耳机",
      }],
    }],
  });

  assertEquals(actions, [
    {
      type: "open_scene",
      sceneId: "scene-a",
      modelId: "model-a",
      ply: "u1/scene-a/output/point_cloud.ply",
      poses: null,
    },
    {
      type: "fly_to_pose",
      sceneId: "scene-a",
      imageName: "frame_01.jpg",
      matrix: [1, 0, 0, 0],
    },
  ]);
});

Deno.test("buildEvidenceFromSpatialResult 会提取 scene 与 matched frame", () => {
  const evidence = buildEvidenceFromSpatialResult({
    success: true,
    intent: {
      original_query: "黑色耳机在哪",
      parsed_search_text: "黑色耳机",
      filter_start: null,
      filter_end: null,
    },
    threshold: 0.5,
    results: [{
      scene_id: "scene-a",
      similarity: 0.91,
      matched_frames: [{
        image_name: "frame_01.jpg",
        similarity: 0.91,
        transform_matrix: [1, 0, 0, 0],
      }],
    }],
  });

  assertEquals(evidence, {
    sceneId: "scene-a",
    similarity: 0.91,
    matchedFrames: [{
      imageName: "frame_01.jpg",
      similarity: 0.91,
      transformMatrix: [1, 0, 0, 0],
    }],
  });
});

Deno.test("runRecallAgent 会按结构化流协议输出关键事件", async () => {
  const mockResult = {
    success: true,
    mode: "spatial_search",
    intent: {
      original_query: "黑色耳机在哪",
      parsed_search_text: "黑色耳机",
      filter_start: null,
      filter_end: null,
    },
    answer: "已找到目标场景。",
    actions: [
      {
        type: "open_model",
        payload: {
          sceneId: "scene-a",
          modelId: "model-a",
          ply: "u1/scene-a/output/point_cloud.ply",
          poses: null,
        },
      },
      {
        type: "fly_to_pose",
        payload: {
          sceneId: "scene-a",
          imageId: "frame_01.jpg",
          matrix: [1, 0, 0, 0],
        },
      },
    ],
    selection: {
      scene_id: "scene-a",
      model_id: "model-a",
      pose_image_id: "frame_01.jpg",
      confidence: 0.91,
      reason: "综合工具检索得分最高",
    },
    viewer_payload: {
      ply: "https://example.com/scene-a.ply",
      poses: "https://example.com/scene-a-poses.json",
      matrix: [1, 0, 0, 0],
      imageId: "frame_01.jpg",
    },
    candidates: [{
      scene_id: "scene-a",
      model_id: "model-a",
      score: 0.91,
      description: "书桌上的黑色耳机",
      pose_image_id: "frame_01.jpg",
    }],
    tool_trace: [],
    asset_context: {
      last_tool_name: null,
      list: null,
      bundle: null,
      comparison: null,
      operation: null,
    },
  } as unknown as Awaited<ReturnType<typeof runSpatialSearchAgent>>;

  const events: Array<{ event: string; data: Record<string, unknown> }> = [];
  const result = await runRecallAgent("黑色耳机在哪", {
    execute: async () => mockResult,
    onEvent: (event) => {
      events.push({
        event: event.event,
        data: event.data as Record<string, unknown>,
      });
    },
  });

  assertEquals(events.map((event) => event.event), [
    "plan",
    "thinking",
    "tool_call",
    "tool_result",
    "thinking",
    "message",
    "done",
  ]);
  assertEquals(events[2]?.data["name"], "run_spatial_search_agent");
  assertEquals(
    (events[2]?.data["args"] as Record<string, unknown>)["executionMode"],
    "preview",
  );
  assertEquals(events[3]?.data["status"], "success");
  assertExists(events[6]?.data["actions"]);
  assertEquals(result.answer, "已找到目标场景。");
  assertEquals(result.actions[0]?.type, "open_scene");
  assertEquals(result.selected_candidate_reason, "综合工具检索得分最高");
});
