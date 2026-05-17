import * as THREE from 'three';
import type { InteractionProfile } from './classifySceneProfile';

export type InteractionMode = 'recall' | 'inspect' | 'freeWalk';

export interface CameraRigConfig {
  camera: THREE.PerspectiveCamera;
  sceneRadius: number;
  profile?: InteractionProfile;
  focusPoint?: THREE.Vector3;
  defaultDistance?: number;
}

const MAX_PITCH_DEG = 82;

function dampAngle(current: number, target: number, alpha: number): number {
  let diff = target - current;
  while (diff > Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;
  return current + diff * alpha;
}

export class BrainDanceCameraRig {
  mode: InteractionMode = 'recall';

  // Scalar yaw / pitch — the single source of truth for camera orientation.
  // 手机竖屏或部分 WebGL 位姿会携带 ±90° 横滚；触控拖动时必须保留 roll，
  // 否则首次交互会把横滚归零，表现为画面瞬间旋转 90°。
  yaw = 0;
  pitch = 0;
  roll = 0;
  targetYaw = 0;
  targetPitch = 0;
  targetRoll = 0;

  position = new THREE.Vector3();
  targetPosition = new THREE.Vector3();

  // 完整保留采集相机四元数，避免欧拉角拆解在竖屏/横滚场景下丢失真实姿态。
  orientation = new THREE.Quaternion();
  targetOrientation = new THREE.Quaternion();

  // Orbit state (used in inspect mode)
  pivot = new THREE.Vector3();
  distance = 2;
  targetDistance = 2;

  // Refs
  camera: THREE.PerspectiveCamera;
  sceneRadius: number;

  // Damping
  lookDamping = 18;
  moveDamping = 14;

  // Interaction quality state
  isInteracting = false;
  private interactionTimer: ReturnType<typeof setTimeout> | null = null;

  // FPS history for adaptive quality
  private fpsHistory: number[] = [];
  private _targetLodScale = 1.0;

  constructor(config: CameraRigConfig) {
    this.camera = config.camera;
    this.sceneRadius = config.sceneRadius;

    if (config.focusPoint) {
      this.pivot.copy(config.focusPoint);
    }
    if (config.defaultDistance) {
      this.distance = config.defaultDistance;
      this.targetDistance = config.defaultDistance;
    }

    if (config.profile === 'object_orbit') {
      this.mode = 'inspect';
    } else {
      this.mode = 'recall';
    }
  }

  /** Seed state from wherever the camera currently is. */
  initFromCamera() {
    this.position.copy(this.camera.position);
    this.targetPosition.copy(this.camera.position);

    this.orientation.copy(this.camera.quaternion);
    this.targetOrientation.copy(this.camera.quaternion);

    const euler = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.yaw = euler.y;
    this.pitch = euler.x;
    this.roll = euler.z;
    this.targetYaw = this.yaw;
    this.targetPitch = this.pitch;
    this.targetRoll = this.roll;
  }

  // ── Input handlers ──────────────────────────────────────────

  onLookDrag(dx: number, dy: number) {
    this.beginInteraction();

    const fovY = THREE.MathUtils.degToRad(this.camera.fov);
    const fovX = 2 * Math.atan(Math.tan(fovY / 2) * this.camera.aspect);

    const vpW = (this.camera as any).renderer?.domElement?.width || window.innerWidth;
    const vpH = (this.camera as any).renderer?.domElement?.height || window.innerHeight;

    const yawPerPixel = fovX / vpW;
    const pitchPerPixel = fovY / vpH;
    const sensitivity = 1.2;

    const deltaYaw = -dx * yawPerPixel * sensitivity;
    const deltaPitch = -dy * pitchPerPixel * sensitivity;

    if (this.mode === 'inspect') {
      this.targetYaw += deltaYaw;
      const maxPitch = THREE.MathUtils.degToRad(85);
      this.targetPitch = THREE.MathUtils.clamp(this.targetPitch + deltaPitch, -maxPitch, maxPitch);
      return;
    }

    // 第一人称/回忆模式直接对完整四元数做增量旋转，避免首次触控把 roll 重建丢失。
    const qYaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), deltaYaw);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.targetOrientation).normalize();
    const qPitch = new THREE.Quaternion().setFromAxisAngle(right, deltaPitch);
    this.targetOrientation.premultiply(qYaw).premultiply(qPitch).normalize();

    const euler = new THREE.Euler().setFromQuaternion(this.targetOrientation, 'YXZ');
    this.targetYaw = euler.y;
    this.targetPitch = euler.x;
    this.targetRoll = euler.z;
  }

  onPinch(scaleDelta: number) {
    this.beginInteraction();
    this.targetDistance *= Math.exp(-scaleDelta * (this.mode === 'inspect' ? 0.002 : 0.001));
    this.targetDistance = THREE.MathUtils.clamp(
      this.targetDistance,
      this.sceneRadius * 0.1,
      this.sceneRadius * 10,
    );
  }

  onPan(dx: number, dy: number) {
    this.beginInteraction();

    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);

    if (this.mode === 'inspect') {
      const speed = this.distance * 0.002;
      this.pivot.add(right.clone().multiplyScalar(-dx * speed));
      this.pivot.add(up.clone().multiplyScalar(dy * speed));
    } else {
      const speed = this.sceneRadius * 0.001;
      this.targetPosition.add(right.clone().multiplyScalar(-dx * speed));
      this.targetPosition.add(up.clone().multiplyScalar(dy * speed));
    }
  }

  // ── Mode switching ──────────────────────────────────────────

  setMode(mode: InteractionMode) {
    if (this.mode === mode) return;
    const prev = this.mode;
    this.mode = mode;

    if (prev !== 'inspect' && mode === 'inspect') {
      // → Orbit: derive pivot from current forward ray
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      this.pivot.copy(this.position).add(fwd.multiplyScalar(this.distance));
      this.targetDistance = this.position.distanceTo(this.pivot);

      const offset = this.position.clone().sub(this.pivot);
      this.targetYaw = Math.atan2(offset.x, offset.z);
      this.targetPitch = Math.asin(
        THREE.MathUtils.clamp(offset.y / offset.length(), -1, 1),
      );
    } else if (prev === 'inspect' && mode !== 'inspect') {
      // → FPS: keep current camera state
      this.targetPosition.copy(this.position);
    }
  }

  focusOnPoint(point: THREE.Vector3) {
    this.pivot.copy(point);
    this.targetDistance = this.position.distanceTo(point);

    const offset = this.position.clone().sub(this.pivot);
    this.targetYaw = Math.atan2(offset.x, offset.z);
    this.targetPitch = Math.asin(
      THREE.MathUtils.clamp(offset.y / offset.length(), -1, 1),
    );

    this.mode = 'inspect';
  }

  // ── Programmatic flight ─────────────────────────────────────

  flyToPose(
    pos: THREE.Vector3,
    quat: THREE.Quaternion,
    onComplete?: () => void,
  ) {
    // 采集镜头跳转必须回到 recall/first-person 模式。
    // 如果仍停留在 inspect/orbit，下一帧会用 lookAt(pivot) 覆盖真实相机姿态，
    // 用户一触控就会看到画面被轨道相机强行转正。
    this.mode = 'recall';
    this.targetOrientation.copy(quat).normalize();
    this.orientation.copy(this.camera.quaternion).normalize();
    const euler = new THREE.Euler().setFromQuaternion(quat, 'YXZ');

    this.targetPosition.copy(pos);
    this.targetYaw = euler.y;
    this.targetPitch = THREE.MathUtils.clamp(
      euler.x,
      -THREE.MathUtils.degToRad(MAX_PITCH_DEG),
      THREE.MathUtils.degToRad(MAX_PITCH_DEG),
    );
    this.targetRoll = euler.z;

    if (onComplete) {
      const dist = this.position.distanceTo(pos);
      const flightMs = Math.min(1200, Math.max(500, (dist / (this.sceneRadius * 2)) * 1000));
      setTimeout(onComplete, flightMs);
    }
  }

  // ── Per-frame update ────────────────────────────────────────

  update(dt: number) {
    if (dt <= 0 || dt > 0.5) dt = 1 / 60;

    const lookAlpha = 1 - Math.exp(-dt * this.lookDamping);
    const moveAlpha = 1 - Math.exp(-dt * this.moveDamping);

    this.yaw = dampAngle(this.yaw, this.targetYaw, lookAlpha);
    this.pitch = THREE.MathUtils.lerp(this.pitch, this.targetPitch, lookAlpha);
    this.roll = dampAngle(this.roll, this.targetRoll, lookAlpha);
    this.orientation.slerp(this.targetOrientation, lookAlpha).normalize();
    this.distance = THREE.MathUtils.lerp(this.distance, this.targetDistance, moveAlpha);
    this.position.lerp(this.targetPosition, moveAlpha);

    if (this.mode === 'freeWalk' || this.mode === 'recall') {
      this.applyFirstPerson();
    } else {
      this.applyOrbit();
    }

    this.camera.updateProjectionMatrix();
  }

  // ── Quality helpers ─────────────────────────────────────────

  getRecommendedLodScale(fps: number): number {
    this.fpsHistory.push(fps);
    if (this.fpsHistory.length > 10) this.fpsHistory.shift();

    const avg =
      this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;

    if (this.isInteracting) {
      this._targetLodScale = 0.65;
    } else if (avg < 25) {
      this._targetLodScale = 0.5;
    } else if (avg < 35) {
      this._targetLodScale = 0.7;
    } else if (avg < 45) {
      this._targetLodScale = 0.85;
    } else {
      this._targetLodScale = 1.0;
    }

    return this._targetLodScale;
  }

  getRecommendedDpr(baseDpr: number): number {
    if (this.isInteracting) return Math.min(baseDpr, 1.0);
    return Math.min(baseDpr, 1.5);
  }

  get targetLodScale() {
    return this._targetLodScale;
  }

  // ── Cleanup ─────────────────────────────────────────────────

  dispose() {
    if (this.interactionTimer !== null) clearTimeout(this.interactionTimer);
  }

  // ── Internal ────────────────────────────────────────────────

  private applyFirstPerson() {
    this.camera.position.copy(this.position);

    this.camera.quaternion.copy(this.orientation).normalize();
  }

  private applyOrbit() {
    const x = Math.sin(this.yaw) * Math.cos(this.pitch) * this.distance;
    const y = Math.sin(this.pitch) * this.distance;
    const z = Math.cos(this.yaw) * Math.cos(this.pitch) * this.distance;

    this.position.set(this.pivot.x + x, this.pivot.y + y, this.pivot.z + z);
    this.camera.position.copy(this.position);
    this.camera.lookAt(this.pivot);

    // lookAt 只决定注视方向，会用 camera.up 重建“正上方”，从而抹掉竖屏位姿的 roll。
    // 对象轨道模式是触控后最常进入的分支，因此这里需要在 lookAt 后把横滚补回来。
    if (Math.abs(this.roll) > 1e-6) {
      this.camera.rotateZ(this.roll);
    }
  }

  private beginInteraction() {
    this.isInteracting = true;
    if (this.interactionTimer !== null) clearTimeout(this.interactionTimer);
    this.interactionTimer = setTimeout(() => {
      this.isInteracting = false;
      this.interactionTimer = null;
    }, 300);
  }
}
