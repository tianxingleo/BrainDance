export const DEFAULT_FOCAL_PX = 380;
export const DEFAULT_SCENE_RADIUS = 1.2;

export const calcFovFromFocal = (focalPx, imageHeightPx) => {
  if (!focalPx || !imageHeightPx) return null;
  return 2 * Math.atan((imageHeightPx / 2) / focalPx) * (180 / Math.PI);
};

export const calcFocalFromFov = (fovDeg, imageHeightPx) => {
  if (!fovDeg || !imageHeightPx) return null;
  const halfFovRad = (fovDeg * Math.PI / 180) / 2;
  if (halfFovRad <= 0) return null;
  return (imageHeightPx / 2) / Math.tan(halfFovRad);
};

export const clampFocalPx = (focalPx, min, max) => {
  if (!Number.isFinite(focalPx)) return null;
  return Math.min(max, Math.max(min, focalPx));
};
