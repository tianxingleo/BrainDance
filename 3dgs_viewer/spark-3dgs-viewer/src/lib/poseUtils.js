import * as THREE from 'three';

export const normalizeMatrixArray = (input) => {
  if (!Array.isArray(input)) return null;

  if (input.length === 16) {
    const flat = input.map((value) => Number(value));
    return flat.every(Number.isFinite) ? flat : null;
  }

  if (input.length === 4 && input.every((row) => Array.isArray(row) && row.length === 4)) {
    const flat = input.flat().map((value) => Number(value));
    return flat.every(Number.isFinite) ? flat : null;
  }

  return null;
};

export const normalizeImageId = (value) => {
  if (value == null) return '';

  let text = String(value).trim();
  if (!text) return '';

  try {
    text = decodeURIComponent(text);
  } catch (_) {}

  text = text.replace(/\\/g, '/');
  const parts = text.split('/');
  return (parts[parts.length - 1] || '').trim().toLowerCase();
};

export const getPoseImageId = (pose) => {
  if (!pose) return '';
  const directId = pose.id || pose.image_id || pose.imageId;
  if (directId) return normalizeImageId(directId);

  if (typeof pose.image_url !== 'string' || pose.image_url.length === 0) return '';
  return normalizeImageId(pose.image_url.split('?')[0]);
};

export const findPoseByInitialTarget = (target, poses) => {
  if (!target || poses.length === 0) return null;

  const targetImageId = normalizeImageId(target.imageId);
  if (targetImageId) {
    const byImage = poses.find((pose) => getPoseImageId(pose) === targetImageId);
    if (byImage) return byImage;
  }

  const targetMatrix = normalizeMatrixArray(target.matrix);
  if (!targetMatrix) return null;

  let bestPose = null;
  let bestDiff = Number.POSITIVE_INFINITY;

  for (const pose of poses) {
    const poseMatrix = normalizeMatrixArray(pose.matrix);
    if (!poseMatrix) continue;

    let maxDiff = 0;
    for (let i = 0; i < 16; i += 1) {
      const diff = Math.abs(poseMatrix[i] - targetMatrix[i]);
      if (diff > maxDiff) maxDiff = diff;
      if (maxDiff >= bestDiff) break;
    }

    if (maxDiff < bestDiff) {
      bestDiff = maxDiff;
      bestPose = pose;
    }
  }

  return bestDiff <= 1e-4 ? bestPose : null;
};

export const parseInitialInputFromUrl = () => {
  const params = new URLSearchParams(window.location.search);
  const payload = params.get('payload');
  if (payload) {
    try {
      const decoded = JSON.parse(decodeURIComponent(payload));
      return {
        ply: decoded.ply || null,
        poses: decoded.poses || null,
        matrix: decoded.matrix || null,
        imageId: decoded.imageId || null,
      };
    } catch (error) {
      console.warn('[SparkViewer] 无法解析 payload 查询参数:', error);
    }
  }

  const ply = params.get('ply');
  const poses = params.get('poses');
  const matrix = params.get('matrix');
  const imageId = params.get('imageId');

  let parsedMatrix = null;
  if (matrix) {
    try {
      parsedMatrix = JSON.parse(decodeURIComponent(matrix));
    } catch (error) {
      console.warn('[SparkViewer] 无法解析 matrix 查询参数:', error);
    }
  }

  if (ply || poses || parsedMatrix || imageId) {
    return {
      ply: ply || null,
      poses: poses || null,
      matrix: parsedMatrix,
      imageId: imageId || null,
    };
  }

  return null;
};

export const resolveImageUrl = (imageUrl, posesUrl) => {
  if (!imageUrl || imageUrl.startsWith('http')) return imageUrl;
  if (!posesUrl || !posesUrl.startsWith('http')) return imageUrl;

  const baseUrl = posesUrl.substring(0, posesUrl.lastIndexOf('/'));
  let relPath = imageUrl;
  const imagesIndex = relPath.indexOf('images/');
  if (imagesIndex !== -1) {
    relPath = relPath.substring(imagesIndex);
  } else if (relPath.startsWith('/models/')) {
    relPath = relPath.substring('/models/'.length);
  } else if (relPath.startsWith('/')) {
    relPath = relPath.substring(1);
  }
  return `${baseUrl}/${relPath}`;
};

export const deriveHighlightPointFromPose = (matrixValues, sceneRadius) => {
  const matrix = new THREE.Matrix4().fromArray(matrixValues);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, quaternion, scale);
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion).normalize();
  return position.clone().add(forward.multiplyScalar(Math.max(sceneRadius * 0.8, 0.4)));
};
