import * as THREE from 'three';
import {
  SplatEdit,
  SplatEditRgbaBlendMode,
  SplatEditSdf,
  SplatEditSdfType,
} from '@sparkjsdev/spark';

export const createSphereHighlightEffect = (sceneRadius, enabled = true) => {
  const edit = new SplatEdit({
    name: 'Search Highlight',
    rgbaBlendMode: SplatEditRgbaBlendMode.ADD_RGBA,
    softEdge: Math.max(sceneRadius * 0.18, 0.08),
    sdfSmooth: Math.max(sceneRadius * 0.12, 0.04),
  });

  const sdf = new SplatEditSdf({
    type: SplatEditSdfType.SPHERE,
    color: new THREE.Color('#d86f3d'),
    opacity: 0.55,
    radius: Math.max(sceneRadius * 0.16, 0.08),
  });

  edit.visible = enabled;
  edit.add(sdf);

  return { edit, sdf };
};

export const updateSphereHighlight = (effect, { enabled, point, radius }) => {
  if (!effect?.edit || !effect?.sdf) return;
  effect.edit.visible = enabled;
  if (!enabled || !point) return;

  effect.sdf.position.copy(point);
  if (radius != null) {
    effect.sdf.radius = radius;
  }
};

export const createClipPlaneEffect = (sceneRadius, enabled = false) => {
  const edit = new SplatEdit({
    name: 'Clip Plane',
    rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY,
    invert: true,
    softEdge: Math.max(sceneRadius * 0.04, 0.02),
    sdfSmooth: Math.max(sceneRadius * 0.02, 0.01),
  });

  const sdf = new SplatEditSdf({
    type: SplatEditSdfType.PLANE,
    color: new THREE.Color(0, 0, 0),
    opacity: 0.0,
  });

  edit.visible = enabled;
  edit.add(sdf);

  return { edit, sdf };
};

export const updateClipPlaneEffect = (effect, { enabled, point, normal }) => {
  if (!effect?.edit || !effect?.sdf) return;
  effect.edit.visible = enabled;
  if (!enabled || !point || !normal) return;

  effect.sdf.position.copy(point);

  const target = point.clone().add(normal);
  effect.sdf.lookAt(target);
};
