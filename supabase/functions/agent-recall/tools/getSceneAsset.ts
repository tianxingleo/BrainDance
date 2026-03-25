type SpatialSearchResponse = {
  selection: {
    scene_id: string | null;
  };
  candidates: Array<{
    scene_id: string;
    score: number;
    pose_image_id: string | null;
  }>;
  viewer_payload: {
    matrix: unknown;
  };
};

export function buildEvidenceFromSpatialResult(
  result: SpatialSearchResponse,
): {
  sceneId: string;
  similarity: number;
  matchedFrames: Array<{
    imageName: string;
    similarity: number;
    transformMatrix: unknown;
  }>;
} | null {
  const sceneId = result.selection.scene_id;
  if (!sceneId) {
    return null;
  }

  const selectedCandidate =
    result.candidates.find((candidate) => candidate.scene_id === sceneId) ??
      result.candidates[0];

  if (!selectedCandidate) {
    return null;
  }

  const matchedFrames = selectedCandidate.pose_image_id
    ? [{
      imageName: selectedCandidate.pose_image_id,
      similarity: selectedCandidate.score,
      transformMatrix: result.viewer_payload.matrix ?? null,
    }]
    : [];

  return {
    sceneId,
    similarity: selectedCandidate.score,
    matchedFrames,
  };
}
