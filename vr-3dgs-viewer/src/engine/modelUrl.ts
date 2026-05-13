export function getVrModelCandidates(modelUrl: string): string[] {
  const candidates = [modelUrl]
  if (/point_cloud\.ply(?:\?.*)?$/i.test(modelUrl)) {
    candidates.unshift(
      modelUrl.replace(/point_cloud\.ply(\?.*)?$/i, 'point_cloud.splat$1'),
      modelUrl.replace(/point_cloud\.ply(\?.*)?$/i, 'point_cloud.ksplat$1'),
    )
  }

  return [...new Set(candidates)]
}
