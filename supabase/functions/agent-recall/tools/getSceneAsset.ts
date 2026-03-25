import type { SearchModelsResponse } from "../../search-models/shared.ts";

function extractSceneLabel(topResult: Record<string, unknown>): string | null {
  const displayName = typeof topResult.display_name === "string"
    ? topResult.display_name.trim()
    : "";
  if (displayName.length > 0) {
    return displayName;
  }
  return typeof topResult.scene_id === "string" ? topResult.scene_id : null;
}

export function buildEvidenceFromSpatialResult(
  result: SearchModelsResponse,
): {
  sceneId: string;
  similarity: number;
  matchedFrames: Array<{
    imageName: string;
    similarity: number;
    transformMatrix: unknown;
  }>;
} | null {
  const topResult = result.results[0];
  if (!topResult) {
    return null;
  }

  const sceneId = extractSceneLabel(topResult);
  if (!sceneId) {
    return null;
  }

  const similarity = Number(topResult.similarity ?? 0);
  const rawFrames = Array.isArray(topResult.matched_frames)
    ? topResult.matched_frames as Array<Record<string, unknown>>
    : [];
  const matchedFrames = rawFrames.map((frame) => ({
    imageName: String(frame.image_name ?? ""),
    similarity: Number(frame.similarity ?? similarity),
    transformMatrix: frame.transform_matrix ?? null,
  })).filter((frame) => frame.imageName.length > 0);

  return {
    sceneId,
    similarity,
    matchedFrames,
  };
}
