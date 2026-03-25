import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import { mapSpatialActionsToRecallActions } from "./tools/buildViewAction.ts";
import { buildEvidenceFromSpatialResult } from "./tools/getSceneAsset.ts";

Deno.test("mapSpatialActionsToRecallActions 会转换为前端稳定动作协议", () => {
  const actions = mapSpatialActionsToRecallActions([
    {
      type: "open_model",
      payload: {
        sceneId: "scene-a",
        modelId: "model-a",
        ply: "https://example.com/model.ply",
        poses: "https://example.com/poses.json",
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
    {
      type: "highlight_hotspot",
      payload: {
        sceneId: "scene-a",
        imageId: "frame_01.jpg",
        label: "黑色耳机",
      },
    },
  ]);

  assertEquals(actions, [
    {
      type: "open_scene",
      sceneId: "scene-a",
      modelId: "model-a",
      ply: "https://example.com/model.ply",
      poses: "https://example.com/poses.json",
    },
    {
      type: "fly_to_pose",
      sceneId: "scene-a",
      imageName: "frame_01.jpg",
      matrix: [1, 0, 0, 0],
    },
    {
      type: "highlight_region",
      sceneId: "scene-a",
      imageName: "frame_01.jpg",
      label: "黑色耳机",
      matrix: null,
    },
  ]);
});

Deno.test("buildEvidenceFromSpatialResult 会提取 scene 与 matched frame", () => {
  const evidence = buildEvidenceFromSpatialResult({
    selection: {
      scene_id: "scene-a",
    },
    candidates: [
      {
        scene_id: "scene-a",
        score: 0.91,
        pose_image_id: "frame_01.jpg",
      },
    ],
    viewer_payload: {
      matrix: [1, 0, 0, 0],
    },
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
