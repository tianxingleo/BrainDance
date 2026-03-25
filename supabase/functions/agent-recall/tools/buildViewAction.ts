import type { SearchModelsResponse } from "../../search-models/shared.ts";
import type { AgentRecallResponse } from "../schemas/response.ts";

function extractSceneLabel(topResult: Record<string, unknown>): string | null {
  const displayName = typeof topResult.display_name === "string"
    ? topResult.display_name.trim()
    : "";
  if (displayName.length > 0) {
    return displayName;
  }
  return typeof topResult.scene_id === "string" ? topResult.scene_id : null;
}

function buildOpenSceneAction(
  topResult: Record<string, unknown>,
): AgentRecallResponse["actions"][number] | null {
  const sceneId = extractSceneLabel(topResult);
  if (!sceneId) {
    return null;
  }

  return {
    type: "open_scene",
    sceneId,
    modelId: typeof topResult.id === "string" ? topResult.id : null,
    ply: typeof topResult.ply_path === "string" ? topResult.ply_path : null,
    poses: null,
  };
}

export function buildRecallActionsFromSearchResult(
  result: SearchModelsResponse,
): AgentRecallResponse["actions"] {
  const mapped: AgentRecallResponse["actions"] = [];
  const topResult = result.results[0];

  if (!topResult || typeof topResult !== "object") {
    return mapped;
  }

  const openAction = buildOpenSceneAction(topResult);
  if (openAction) {
    mapped.push(openAction);
  }

  const sceneId = extractSceneLabel(topResult);
  const rawFrames = Array.isArray(topResult.matched_frames)
    ? topResult.matched_frames as Array<Record<string, unknown>>
    : [];
  const bestFrame = rawFrames[0];

  if (sceneId && bestFrame) {
    const imageName = typeof bestFrame.image_name === "string"
      ? bestFrame.image_name
      : undefined;
    const matrix = bestFrame.transform_matrix ?? null;
    mapped.push({
      type: "fly_to_pose",
      sceneId,
      imageName,
      matrix,
    });
  }

  return mapped;
}
