import * as THREE from 'three';

export type InteractionProfile = 'object_orbit' | 'panorama_fps' | 'walkthrough' | 'hybrid';

export interface ProfileResult {
  profile: InteractionProfile;
  confidence: number;
  focusPoint?: number[];
  defaultRadius?: number;
}

interface PoseSample {
  position: THREE.Vector3;
  forward: THREE.Vector3;
  quaternion: THREE.Quaternion;
}

function normalizeMatrixArray(input: unknown): number[] | null {
  if (!Array.isArray(input)) return null;
  if (input.length === 16) {
    const flat = input.map((v: unknown) => Number(v));
    return flat.every(Number.isFinite) ? flat : null;
  }
  return null;
}

export function extractPoseData(poses: Array<{ matrix: unknown }>): PoseSample[] {
  return poses
    .map((pose) => {
      const matrixValues = normalizeMatrixArray(pose.matrix);
      if (!matrixValues) return null;

      const m = new THREE.Matrix4().fromArray(matrixValues);
      const position = new THREE.Vector3();
      const quaternion = new THREE.Quaternion();
      const scale = new THREE.Vector3();
      m.decompose(position, quaternion, scale);

      const forward = new THREE.Vector3(0, 0, -1)
        .applyQuaternion(quaternion)
        .normalize();

      return { position, forward, quaternion };
    })
    .filter(Boolean) as PoseSample[];
}

/**
 * Estimate the common focus point where camera rays converge.
 * Solves: minimize Σ ||(I - f_i f_i^T)(C - P_i)||^2
 */
export function estimateRayFocus(samples: PoseSample[]): THREE.Vector3 {
  const M = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const rhs = [0, 0, 0];

  for (const { position: p, forward: f } of samples) {
    const x = f.x,
      y = f.y,
      z = f.z;

    const IminusFF = [
      [1 - x * x, -x * y, -x * z],
      [-y * x, 1 - y * y, -y * z],
      [-z * x, -z * y, 1 - z * z],
    ];

    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 3; c += 1) {
        M[r][c] += IminusFF[r][c];
      }
      rhs[r] +=
        IminusFF[r][0] * p.x +
        IminusFF[r][1] * p.y +
        IminusFF[r][2] * p.z;
    }
  }

  const mat = new THREE.Matrix3().set(
    M[0][0], M[0][1], M[0][2],
    M[1][0], M[1][1], M[1][2],
    M[2][0], M[2][1], M[2][2],
  );

  try {
    const inv = mat.clone().invert();
    return new THREE.Vector3(rhs[0], rhs[1], rhs[2]).applyMatrix3(inv);
  } catch {
    const centroid = new THREE.Vector3();
    for (const s of samples) centroid.add(s.position);
    centroid.divideScalar(Math.max(samples.length, 1));
    return centroid;
  }
}

/**
 * Measure horizontal angular coverage by bucketing forward azimuths.
 * Returns 0..1 where 1 = full 360° coverage.
 */
export function estimateForwardCoverage(samples: PoseSample[], bucketCount = 12): number {
  const buckets = new Set<number>();

  for (const { forward } of samples) {
    const angle = Math.atan2(forward.z, forward.x);
    const normalized = (angle + Math.PI) / (Math.PI * 2);
    const bucket = Math.floor(normalized * bucketCount);
    buckets.add(bucket);
  }

  return buckets.size / bucketCount;
}

/**
 * Classify scene interaction profile from pose data.
 * Determines whether the camera topology suggests:
 * - object_orbit: cameras looking at a common center (scanning an object)
 * - panorama_fps: cameras clustered, looking in many directions (standing in place)
 * - walkthrough: cameras spread along a path (walking through space)
 * - hybrid: unclear, let user decide
 */
export function classifyInteractionProfile(
  poses: Array<{ matrix: unknown }>,
  sceneRadius: number,
): ProfileResult {
  const samples = extractPoseData(poses);

  if (samples.length < 4) {
    return { profile: 'hybrid', confidence: 0.3 };
  }

  const focus = estimateRayFocus(samples);

  const positions = samples.map((s) => s.position);
  const meanPosition = positions
    .reduce((acc, p) => acc.add(p), new THREE.Vector3())
    .multiplyScalar(1 / positions.length);

  const distancesToFocus = samples.map((s) => s.position.distanceTo(focus));
  const meanRadius =
    distancesToFocus.reduce((a, b) => a + b, 0) / distancesToFocus.length;

  const inwardScores = samples.map((s) => {
    const toFocus = focus.clone().sub(s.position).normalize();
    return s.forward.dot(toFocus);
  });
  const inwardMean = inwardScores.reduce((a, b) => a + b, 0) / inwardScores.length;

  const positionSpread = Math.sqrt(
    positions
      .map((p) => p.distanceToSquared(meanPosition))
      .reduce((a, b) => a + b, 0) / positions.length,
  );

  const forwardCoverage = estimateForwardCoverage(samples);
  const positionSpreadNorm = positionSpread / Math.max(sceneRadius, 1e-3);

  if (inwardMean > 0.65 && meanRadius > sceneRadius * 0.35) {
    return {
      profile: 'object_orbit',
      confidence: THREE.MathUtils.clamp((inwardMean - 0.55) / 0.35, 0, 1),
      focusPoint: focus.toArray(),
      defaultRadius: meanRadius,
    };
  }

  if (positionSpreadNorm < 0.25 && forwardCoverage > 0.55) {
    return {
      profile: 'panorama_fps',
      confidence: 0.75,
      focusPoint: meanPosition.toArray(),
    };
  }

  if (positionSpreadNorm >= 0.25) {
    return {
      profile: 'walkthrough',
      confidence: 0.65,
      focusPoint: meanPosition.toArray(),
    };
  }

  return {
    profile: 'hybrid',
    confidence: 0.4,
    focusPoint: focus.toArray(),
  };
}
