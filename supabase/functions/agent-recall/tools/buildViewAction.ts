import type { AgentRecallResponse } from "../schemas/response.ts";

type SpatialAction = {
  type: "open_model" | "fly_to_pose" | "highlight_hotspot";
  payload?: Record<string, unknown>;
};

export function mapSpatialActionsToRecallActions(
  actions: SpatialAction[],
): AgentRecallResponse["actions"] {
  const mapped: AgentRecallResponse["actions"] = [];

  for (const action of actions) {
    const payload = action.payload ?? {};
    const sceneId = typeof payload.sceneId === "string"
      ? payload.sceneId
      : null;
    if (!sceneId) continue;

    if (action.type === "open_model") {
      mapped.push({
        type: "open_scene" as const,
        sceneId,
        modelId: typeof payload.modelId === "string" ? payload.modelId : null,
        ply: typeof payload.ply === "string" ? payload.ply : null,
        poses: typeof payload.poses === "string" ? payload.poses : null,
      });
      continue;
    }

    if (action.type === "fly_to_pose") {
      mapped.push({
        type: "fly_to_pose" as const,
        sceneId,
        imageName: typeof payload.imageId === "string"
          ? payload.imageId
          : undefined,
        matrix: payload.matrix ?? null,
      });
      continue;
    }

    mapped.push({
      type: "highlight_region" as const,
      sceneId,
      imageName: typeof payload.imageId === "string"
        ? payload.imageId
        : undefined,
      label: typeof payload.label === "string" ? payload.label : undefined,
      matrix: payload.matrix ?? null,
    });
  }

  return mapped;
}
