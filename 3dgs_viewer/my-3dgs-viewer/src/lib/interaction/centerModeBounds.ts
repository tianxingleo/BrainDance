import * as THREE from 'three';

export type CenterModeTopology =
  | 'full_orbit'
  | 'semi_orbit'
  | 'panorama_anchor'
  | 'walkthrough';

export interface AngleInterval {
  min: number;
  max: number;
}

export interface CenterModeBounds {
  topology: CenterModeTopology;
  center: THREE.Vector3;
  radiusMin: number;
  radiusMax: number;
  radiusElastic: number;
  yawIntervals: AngleInterval[];
  yawElastic: number;
  pitchMin: number;
  pitchMax: number;
  pitchElastic: number;
}

interface PoseSample {
  position: THREE.Vector3;
  forward: THREE.Vector3;
}

const TAU = Math.PI * 2;

function normalizeAngle(angle: number): number {
  let value = angle % TAU;
  if (value < 0) value += TAU;
  return value;
}

function wrapAngle(angle: number): number {
  let value = angle;
  while (value <= -Math.PI) value += TAU;
  while (value > Math.PI) value -= TAU;
  return value;
}

function shortestAngleDiff(from: number, to: number): number {
  return wrapAngle(to - from);
}

function vectorMean(vectors: THREE.Vector3[]): THREE.Vector3 {
  const out = new THREE.Vector3();
  if (vectors.length === 0) return out;
  for (const vector of vectors) out.add(vector);
  return out.multiplyScalar(1 / vectors.length);
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  if (values.length === 1) return values[0]!;

  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * THREE.MathUtils.clamp(p, 0, 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower]!;

  const t = index - lower;
  return sorted[lower]! * (1 - t) + sorted[upper]! * t;
}

function estimateFocusPoint(samples: PoseSample[]): THREE.Vector3 {
  const matrix = new THREE.Matrix3().set(
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
  );
  const rhs = new THREE.Vector3();

  for (const sample of samples) {
    const f = sample.forward.clone().normalize();
    const p = sample.position;
    const x = f.x;
    const y = f.y;
    const z = f.z;

    matrix.elements[0] += 1 - x * x;
    matrix.elements[1] += -x * y;
    matrix.elements[2] += -x * z;
    matrix.elements[3] += -y * x;
    matrix.elements[4] += 1 - y * y;
    matrix.elements[5] += -y * z;
    matrix.elements[6] += -z * x;
    matrix.elements[7] += -z * y;
    matrix.elements[8] += 1 - z * z;

    rhs.x += (1 - x * x) * p.x + (-x * y) * p.y + (-x * z) * p.z;
    rhs.y += (-y * x) * p.x + (1 - y * y) * p.y + (-y * z) * p.z;
    rhs.z += (-z * x) * p.x + (-z * y) * p.y + (1 - z * z) * p.z;
  }

  try {
    const inv = matrix.clone().invert();
    return rhs.applyMatrix3(inv);
  } catch {
    return vectorMean(samples.map((sample) => sample.position));
  }
}

function parseSamples(
  poses: Array<{ matrix: unknown }>,
): PoseSample[] {
  return poses
    .map((pose) => {
      if (!Array.isArray(pose.matrix) || pose.matrix.length !== 16) return null;
      const matrixValues = pose.matrix.map((value: unknown) => Number(value));
      if (!matrixValues.every(Number.isFinite)) return null;

      const matrix = new THREE.Matrix4().fromArray(matrixValues);
      const position = new THREE.Vector3();
      const quaternion = new THREE.Quaternion();
      const scale = new THREE.Vector3();
      matrix.decompose(position, quaternion, scale);

      const forward = new THREE.Vector3(0, 0, -1)
        .applyQuaternion(quaternion)
        .normalize();

      return { position, forward };
    })
    .filter(Boolean) as PoseSample[];
}

function buildCircularInterval(angles: number[], marginRad: number): AngleInterval[] {
  if (angles.length < 3) return [];

  const sorted = angles.map(normalizeAngle).sort((a, b) => a - b);
  let largestGap = -1;
  let splitIndex = 0;

  for (let i = 0; i < sorted.length; i += 1) {
    const current = sorted[i]!;
    const next = sorted[(i + 1) % sorted.length]!;
    const gap = i === sorted.length - 1 ? next + TAU - current : next - current;
    if (gap > largestGap) {
      largestGap = gap;
      splitIndex = (i + 1) % sorted.length;
    }
  }

  const coverage = TAU - largestGap;
  if (coverage >= THREE.MathUtils.degToRad(330)) return [];

  const min = wrapAngle(sorted[splitIndex]! - marginRad);
  const max = wrapAngle(sorted[(splitIndex + sorted.length - 1) % sorted.length]! + marginRad);
  return [{ min, max }];
}

function isAngleInsideInterval(angle: number, interval: AngleInterval): boolean {
  const value = normalizeAngle(angle);
  const min = normalizeAngle(interval.min);
  const max = normalizeAngle(interval.max);

  if (min <= max) return value >= min && value <= max;
  return value >= min || value <= max;
}

function clampAngleToIntervals(angle: number, intervals: AngleInterval[], elastic: number): number {
  if (intervals.length === 0) return wrapAngle(angle);
  for (const interval of intervals) {
    if (isAngleInsideInterval(angle, interval)) return wrapAngle(angle);
  }

  let nearestBoundary = intervals[0]!.min;
  let bestDistance = Infinity;
  for (const interval of intervals) {
    for (const boundary of intervalBoundaries(interval)) {
      const distance = Math.abs(shortestAngleDiff(angle, boundary));
      if (distance < bestDistance) {
        bestDistance = distance;
        nearestBoundary = boundary;
      }
    }
  }

  const diff = shortestAngleDiff(nearestBoundary, angle);
  const band = Math.max(elastic, THREE.MathUtils.degToRad(2));
  const pulled = diff / (1 + Math.abs(diff) / band);
  return wrapAngle(nearestBoundary + pulled);
}

function intervalBoundaries(interval: AngleInterval): number[] {
  return [interval.min, interval.max];
}

function softClamp(value: number, min: number, max: number, elastic: number): number {
  if (min > max) return value;
  if (value < min) {
    const delta = min - value;
    return min - (delta / (1 + delta / Math.max(elastic, 1e-4)));
  }
  if (value > max) {
    const delta = value - max;
    return max + (delta / (1 + delta / Math.max(elastic, 1e-4)));
  }
  return value;
}

export function buildCenterModeBounds(
  poses: Array<{ matrix: unknown }>,
  sceneCenter: THREE.Vector3,
  sceneRadius: number,
): CenterModeBounds {
  const samples = parseSamples(poses);
  if (samples.length < 3) {
    return {
      topology: 'walkthrough',
      center: sceneCenter.clone(),
      radiusMin: Math.max(sceneRadius * 0.12, 0.08),
      radiusMax: Math.max(sceneRadius * 4, 1.2),
      radiusElastic: Math.max(sceneRadius * 0.08, 0.08),
      yawIntervals: [],
      yawElastic: THREE.MathUtils.degToRad(16),
      pitchMin: THREE.MathUtils.degToRad(-82),
      pitchMax: THREE.MathUtils.degToRad(82),
      pitchElastic: THREE.MathUtils.degToRad(10),
    };
  }

  const focus = estimateFocusPoint(samples);
  const center = sceneCenter.clone().lerp(focus, 0.4);
  const positions = samples.map((sample) => sample.position);
  const forwards = samples.map((sample) => sample.forward);
  const offsets = positions.map((position) => position.clone().sub(center));
  const radii = offsets.map((offset) => offset.length());
  const positionMean = vectorMean(positions);

  const meanRadius = radii.reduce((sum, value) => sum + value, 0) / radii.length;
  const radiusVariance = radii.reduce((sum, value) => sum + ((value - meanRadius) ** 2), 0) / radii.length;
  const radiusStd = Math.sqrt(radiusVariance);
  const radiusCv = meanRadius > 1e-6 ? radiusStd / meanRadius : 0;
  const positionSpread = Math.sqrt(
    positions.reduce((sum, position) => sum + position.distanceToSquared(positionMean), 0) / positions.length,
  );
  const positionSpreadNorm = positionSpread / Math.max(sceneRadius, 1e-3);

  let inwardScore = 0;
  for (const sample of samples) {
    const toCenter = center.clone().sub(sample.position).normalize();
    inwardScore += sample.forward.dot(toCenter);
  }
  inwardScore /= samples.length;

  // 中心模式采用 Z 轴朝上：yaw 在 XY 平面内转，pitch 控制 Z 轴高度。
  const orbitYawAngles = offsets.map((offset) => Math.atan2(offset.y, offset.x));
  const orbitPitchAngles = offsets.map((offset) => {
    const horizontal = Math.sqrt(offset.x * offset.x + offset.y * offset.y);
    return Math.atan2(offset.z, horizontal);
  });
  const forwardYawAngles = forwards.map((forward) => Math.atan2(forward.y, forward.x));
  const forwardPitchAngles = forwards.map((forward) => {
    const horizontal = Math.sqrt(forward.x * forward.x + forward.y * forward.y);
    return Math.atan2(forward.z, horizontal);
  });

  let topology: CenterModeTopology = 'walkthrough';
  if (inwardScore > 0.62 && radiusCv < 0.32) {
    topology = 'full_orbit';
  } else if (inwardScore > 0.42 && positionSpreadNorm < 0.9) {
    topology = 'semi_orbit';
  } else if (positionSpreadNorm < 0.28) {
    topology = 'panorama_anchor';
  }

  const useOrbitAngles = topology === 'full_orbit' || topology === 'semi_orbit';
  const useForwardAngles = topology === 'panorama_anchor';
  const yawSource = useOrbitAngles ? orbitYawAngles : useForwardAngles ? forwardYawAngles : orbitYawAngles;
  const pitchSource = useOrbitAngles ? orbitPitchAngles : useForwardAngles ? forwardPitchAngles : forwardPitchAngles;

  const yawIntervals = buildCircularInterval(
    yawSource,
    THREE.MathUtils.degToRad(topology === 'full_orbit' ? 10 : topology === 'semi_orbit' ? 18 : 26),
  );

  const pitchMarginDeg = topology === 'full_orbit' ? 10 : topology === 'semi_orbit' ? 14 : 18;
  const pitchMin = THREE.MathUtils.clamp(
    percentile(pitchSource, 0.05) - THREE.MathUtils.degToRad(pitchMarginDeg),
    -Math.PI / 2 + 0.03,
    Math.PI / 2 - 0.03,
  );
  const pitchMax = THREE.MathUtils.clamp(
    percentile(pitchSource, 0.95) + THREE.MathUtils.degToRad(pitchMarginDeg),
    -Math.PI / 2 + 0.03,
    Math.PI / 2 - 0.03,
  );

  const radiusMin = Math.max(
    percentile(radii, 0.1) - Math.max(sceneRadius * 0.08, radiusStd * 0.28),
    sceneRadius * 0.08,
  );
  const radiusMax = Math.max(
    percentile(radii, 0.9) + Math.max(sceneRadius * 0.08, radiusStd * 0.42),
    radiusMin + Math.max(sceneRadius * 0.12, 0.15),
  );

  return {
    topology,
    center,
    radiusMin,
    radiusMax,
    radiusElastic: Math.max(sceneRadius * 0.08, radiusStd * 0.18, 0.08),
    yawIntervals,
    yawElastic: THREE.MathUtils.degToRad(topology === 'full_orbit' ? 10 : 14),
    pitchMin,
    pitchMax,
    pitchElastic: THREE.MathUtils.degToRad(topology === 'full_orbit' ? 8 : 12),
  };
}

export function clampCenterModeRadius(value: number, bounds: CenterModeBounds): number {
  return softClamp(value, bounds.radiusMin, bounds.radiusMax, bounds.radiusElastic);
}

export function clampCenterModePitch(value: number, bounds: CenterModeBounds): number {
  return softClamp(value, bounds.pitchMin, bounds.pitchMax, bounds.pitchElastic);
}

export function clampCenterModeYaw(value: number, bounds: CenterModeBounds): number {
  return clampAngleToIntervals(value, bounds.yawIntervals, bounds.yawElastic);
}
