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
  up: THREE.Vector3;
  basisX: THREE.Vector3;
  basisY: THREE.Vector3;
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
  up: THREE.Vector3;
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

function projectOntoPlane(vector: THREE.Vector3, normal: THREE.Vector3): THREE.Vector3 {
  return vector.clone().sub(normal.clone().multiplyScalar(vector.dot(normal)));
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

function buildOrthonormalBasis(
  upHint: THREE.Vector3,
  samples: PoseSample[],
  center: THREE.Vector3,
): { up: THREE.Vector3; basisX: THREE.Vector3; basisY: THREE.Vector3 } {
  const up = upHint.clone().normalize();
  let basisX = vectorMean(
    samples.map((sample) => projectOntoPlane(sample.position.clone().sub(center), up)),
  );
  basisX = projectOntoPlane(basisX, up);
  if (basisX.lengthSq() < 1e-8) {
    basisX = projectOntoPlane(samples[0]!.position.clone().sub(center), up);
  }
  if (basisX.lengthSq() < 1e-8) {
    basisX = new THREE.Vector3(1, 0, 0).projectOnPlane(up);
  }
  if (basisX.lengthSq() < 1e-8) {
    basisX = new THREE.Vector3(0, 0, 1);
  }
  basisX.normalize();

  const basisY = up.clone().cross(basisX).normalize();
  if (basisY.lengthSq() < 1e-8) {
    basisX.set(1, 0, 0).projectOnPlane(up).normalize();
    basisY.copy(up).cross(basisX).normalize();
  }

  return { up, basisX, basisY };
}

function measureCircularCoverage(angles: number[]): number {
  if (angles.length < 2) return 0;
  const sorted = angles.map(normalizeAngle).sort((a, b) => a - b);
  let largestGap = 0;
  for (let i = 0; i < sorted.length; i += 1) {
    const current = sorted[i]!;
    const next = sorted[(i + 1) % sorted.length]!;
    const gap = i === sorted.length - 1 ? next + TAU - current : next - current;
    largestGap = Math.max(largestGap, gap);
  }
  return THREE.MathUtils.clamp((TAU - largestGap) / TAU, 0, 1);
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
      const up = new THREE.Vector3(0, 1, 0)
        .applyQuaternion(quaternion)
        .normalize();

      return { position, forward, up };
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
    const fallbackUp = new THREE.Vector3(0, 1, 0);
    return {
      topology: 'walkthrough',
      center: sceneCenter.clone(),
      up: fallbackUp.clone(),
      basisX: new THREE.Vector3(1, 0, 0),
      basisY: new THREE.Vector3(0, 0, -1),
      radiusMin: Math.max(sceneRadius * 0.06, 0.05),
      radiusMax: Math.max(sceneRadius * 5.5, 1.6),
      radiusElastic: Math.max(sceneRadius * 0.12, 0.12),
      yawIntervals: [],
      yawElastic: THREE.MathUtils.degToRad(16),
      pitchMin: THREE.MathUtils.degToRad(-82),
      pitchMax: THREE.MathUtils.degToRad(82),
      pitchElastic: THREE.MathUtils.degToRad(10),
    };
  }

  const positions = samples.map((sample) => sample.position);
  const forwards = samples.map((sample) => sample.forward);
  const upHint = vectorMean(samples.map((sample) => sample.up));
  const positionMean = vectorMean(positions);
  const positionSpread = Math.sqrt(
    positions.reduce((sum, position) => sum + position.distanceToSquared(positionMean), 0) / positions.length,
  );
  const positionSpreadNorm = positionSpread / Math.max(sceneRadius, 1e-3);

  const center = sceneCenter.clone();
  const orbitFrame = buildOrthonormalBasis(
    upHint.lengthSq() > 1e-8 ? upHint : new THREE.Vector3(0, 1, 0),
    samples,
    center,
  );

  const offsets = positions.map((position) => position.clone().sub(center));
  const radii = offsets.map((offset) => offset.length());
  const meanRadius = radii.reduce((sum, value) => sum + value, 0) / radii.length;
  const radiusVariance = radii.reduce((sum, value) => sum + ((value - meanRadius) ** 2), 0) / radii.length;
  const radiusStd = Math.sqrt(radiusVariance);
  const radiusCv = meanRadius > 1e-6 ? radiusStd / meanRadius : 0;

  let inwardScore = 0;
  for (const sample of samples) {
    const toCenter = center.clone().sub(sample.position).normalize();
    inwardScore += sample.forward.dot(toCenter);
  }
  inwardScore /= samples.length;

  const orbitYawAngles = offsets.map((offset) => {
    const x = offset.dot(orbitFrame.basisX);
    const y = offset.dot(orbitFrame.basisY);
    return Math.atan2(y, x);
  });
  const orbitPitchAngles = offsets.map((offset) => {
    const z = offset.dot(orbitFrame.up);
    const horizontal = Math.sqrt(Math.max(offset.lengthSq() - (z * z), 0));
    return Math.atan2(z, horizontal);
  });
  const forwardYawAngles = forwards.map((forward) => {
    const x = forward.dot(orbitFrame.basisX);
    const y = forward.dot(orbitFrame.basisY);
    return Math.atan2(y, x);
  });
  const forwardPitchAngles = forwards.map((forward) => {
    const z = forward.dot(orbitFrame.up);
    const horizontal = Math.sqrt(Math.max(forward.lengthSq() - (z * z), 0));
    return Math.atan2(z, horizontal);
  });

  const yawCoverage = measureCircularCoverage(orbitYawAngles);
  let topology: CenterModeTopology = 'walkthrough';
  if (inwardScore > 0.64 && yawCoverage > 0.82 && radiusCv < 0.38) {
    topology = 'full_orbit';
  } else if (inwardScore > 0.4 && yawCoverage > 0.28 && positionSpreadNorm < 0.95) {
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
    THREE.MathUtils.degToRad(topology === 'full_orbit' ? 10 : topology === 'semi_orbit' ? 16 : 22),
  );

  const pitchMarginDeg = topology === 'full_orbit' ? 10 : topology === 'semi_orbit' ? 14 : 16;
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

  const radiusLowerPct = topology === 'full_orbit' ? 0.01 : topology === 'semi_orbit' ? 0.02 : 0.04;
  const radiusUpperPct = topology === 'full_orbit' ? 0.98 : topology === 'semi_orbit' ? 0.96 : 0.94;
  const lowerPadding = Math.max(sceneRadius * 0.015, radiusStd * 0.14, 0.02);
  const upperPadding = Math.max(sceneRadius * 0.08, radiusStd * 0.42, 0.06);
  const radiusMin = Math.max(
    percentile(radii, radiusLowerPct) - lowerPadding,
    sceneRadius * 0.015,
  );
  const radiusMax = Math.max(
    percentile(radii, radiusUpperPct) + upperPadding,
    radiusMin + Math.max(sceneRadius * 0.10, 0.12),
  );
  const radiusSpan = Math.max(radiusMax - radiusMin, 0.12);

  return {
    topology,
    center,
    up: orbitFrame.up,
    basisX: orbitFrame.basisX,
    basisY: orbitFrame.basisY,
    radiusMin,
    radiusMax,
    radiusElastic: Math.max(radiusSpan * 0.28, sceneRadius * 0.05, radiusStd * 0.16, 0.06),
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
