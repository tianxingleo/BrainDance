import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
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
