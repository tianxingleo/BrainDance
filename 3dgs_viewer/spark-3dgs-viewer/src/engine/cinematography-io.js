import * as THREE from 'three';
import gsap from 'gsap';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { FIELD_PRESETS } from './vector-fields.js';

function cubicBezier(p0, p1, p2, p3, t) {
  const k = 1 - t;
  return k * k * k * p0 + 3 * k * k * t * p1 + 3 * k * t * t * p2 + t * t * t * p3;
}

function buildSequenceOffsets(radius) {
  const r = Math.max(radius, 0.75);
  return [
    [new THREE.Vector3(0.0, 0.15, 1.4), new THREE.Vector3(0.8, 0.25, 1.1), new THREE.Vector3(1.2, -0.1, 0.5)],
    [new THREE.Vector3(-1.4, 0.2, 0.1), new THREE.Vector3(-1.0, 0.4, 1.0), new THREE.Vector3(-0.3, 0.1, 1.4)],
    [new THREE.Vector3(0.4, 1.2, 0.6), new THREE.Vector3(-0.4, 1.4, -0.2), new THREE.Vector3(-1.1, 0.9, -0.8)],
    [new THREE.Vector3(0.0, -0.7, 1.5), new THREE.Vector3(1.1, -0.4, 0.8), new THREE.Vector3(1.4, 0.1, -0.3)],
    [new THREE.Vector3(1.4, 0.4, -0.1), new THREE.Vector3(0.8, 0.7, -1.0), new THREE.Vector3(-0.1, 0.2, -1.4)],
    [new THREE.Vector3(-0.6, 0.9, 1.2), new THREE.Vector3(0.3, 1.3, 0.7), new THREE.Vector3(1.0, 0.7, -0.4)],
    [new THREE.Vector3(1.2, -0.3, 1.0), new THREE.Vector3(0.5, -0.7, 1.4), new THREE.Vector3(-0.5, -0.3, 1.1)],
    [new THREE.Vector3(-1.1, 0.6, -0.8), new THREE.Vector3(-0.7, 0.1, -1.3), new THREE.Vector3(0.2, -0.2, -1.5)],
    [new THREE.Vector3(0.9, 1.0, 0.1), new THREE.Vector3(0.1, 1.5, 0.8), new THREE.Vector3(-0.9, 0.8, 0.9)],
    [new THREE.Vector3(-0.2, -0.9, -1.4), new THREE.Vector3(0.6, -0.8, -1.0), new THREE.Vector3(1.2, -0.3, -0.2)],
    [new THREE.Vector3(1.5, 0.1, 0.6), new THREE.Vector3(1.1, 0.5, -0.5), new THREE.Vector3(0.2, 0.8, -1.2)],
    [new THREE.Vector3(-1.4, -0.2, 0.5), new THREE.Vector3(-0.9, 0.6, 0.9), new THREE.Vector3(-0.3, 1.0, 0.0)],
    [new THREE.Vector3(0.1, 0.5, 1.7), new THREE.Vector3(-0.7, 0.2, 1.3), new THREE.Vector3(-1.3, -0.2, 0.7)],
    [new THREE.Vector3(0.7, -1.0, 0.8), new THREE.Vector3(-0.1, -1.2, 0.2), new THREE.Vector3(-1.0, -0.7, -0.5)],
    [new THREE.Vector3(1.0, 0.3, -1.1), new THREE.Vector3(0.2, 0.9, -1.5), new THREE.Vector3(-0.8, 0.4, -1.2)],
  ].map((sequence) => sequence.map((v) => v.multiplyScalar(r)));
}

export class CinematographyEngine {
  constructor({ camera, controls, sceneRadius, sceneCenter, poses, onFocusChange, onSequenceChange }) {
    this.camera = camera;
    this.controls = controls;
    this.sceneRadius = sceneRadius;
    this.sceneCenter = sceneCenter.clone();
    this.poses = poses;
    this.onFocusChange = onFocusChange;
    this.onSequenceChange = onSequenceChange;
    this.focus = sceneCenter.clone();
    this.focusLerp = sceneCenter.clone();
    this.sequenceIndex = 0;
    this.sequenceTime = 0;
    this.autoPlay = true;
    this.bounds = {
      min: Math.max(sceneRadius * 0.9, 0.8),
      max: Math.max(sceneRadius * 4.8, 4.0),
    };
    this.sequences = buildSequenceOffsets(sceneRadius);
    this.userIntent = new THREE.Vector3(0, 0, 1);
    this.baseUp = new THREE.Vector3(0, 1, 0);
    this.manualCameraState = null;
    this.isPoseTransitioning = false;
  }

  setSceneBounds({ radius, center }) {
    this.sceneRadius = radius;
    this.sceneCenter.copy(center);
    this.focus.copy(center);
    this.focusLerp.copy(center);
    this.manualCameraState = null;
    this.isPoseTransitioning = false;
    this.bounds.min = Math.max(radius * 0.9, 0.8);
    this.bounds.max = Math.max(radius * 4.8, 4.0);
    this.sequences = buildSequenceOffsets(radius);
  }

  frameFocus(focus = this.focus) {
    this.manualCameraState = null;
    this.isPoseTransitioning = false;
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const fit = Math.max(this.sceneRadius * 1.2 / Math.tan(fov * 0.5), this.bounds.min);
    const distance = THREE.MathUtils.clamp(fit, this.bounds.min, this.bounds.max);
    const offset = new THREE.Vector3(0, this.sceneRadius * 0.25, distance);
    this.camera.position.copy(focus).add(offset);
    this.camera.lookAt(focus);
  }

  setFocus(point, immediate = false) {
    this.manualCameraState = null;
    this.isPoseTransitioning = false;
    this.focus.copy(point);
    if (immediate) this.focusLerp.copy(point);
    this.onFocusChange?.(this.focus);
  }

  fitDistance(radiusMultiplier = 1.0) {
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const radius = this.sceneRadius * radiusMultiplier;
    const fit = radius / Math.max(Math.sin(fov * 0.5), 0.25);
    return THREE.MathUtils.clamp(fit, this.bounds.min, this.bounds.max);
  }

  _basis() {
    const currentDir = new THREE.Vector3().subVectors(this.camera.position, this.focusLerp).normalize();
    this.userIntent.lerp(currentDir, 0.18).normalize();
    const right = new THREE.Vector3().crossVectors(this.baseUp, this.userIntent).normalize();
    const up = new THREE.Vector3().crossVectors(this.userIntent, right).normalize();
    return { right, up, forward: this.userIntent.clone() };
  }

  _relativeOffset(localOffset) {
    const { right, up, forward } = this._basis();
    return new THREE.Vector3()
      .addScaledVector(right, localOffset.x)
      .addScaledVector(up, localOffset.y)
      .addScaledVector(forward, localOffset.z);
  }

  update(dt) {
    this.focusLerp.lerp(this.focus, 1.0 - Math.exp(-dt * 4.0));
    if (!this.autoPlay) {
      if (this.isPoseTransitioning) {
        return;
      }
      if (this.manualCameraState) {
        this.camera.position.copy(this.manualCameraState.position);
        this.camera.quaternion.copy(this.manualCameraState.quaternion);
      } else {
        this.camera.lookAt(this.focusLerp);
      }
      return;
    }
    this.manualCameraState = null;
    this.sequenceTime += dt;
    const sequence = this.sequences[this.sequenceIndex] || this.sequences[0];
    const phase = (this.sequenceTime % 12.0) / 12.0;
    const a = sequence[0];
    const b = sequence[1];
    const c = sequence[2];
    const d = sequence[0];
    const localX = cubicBezier(a.x, b.x, c.x, d.x, phase);
    const localY = cubicBezier(a.y, b.y, c.y, d.y, phase);
    const localZ = cubicBezier(a.z, b.z, c.z, d.z, phase);
    const offset = this._relativeOffset(new THREE.Vector3(localX, localY, localZ));
    const distance = THREE.MathUtils.clamp(offset.length(), this.bounds.min, this.bounds.max);
    offset.setLength(distance);
    const targetPos = this.focusLerp.clone().add(offset);
    this.camera.position.lerp(targetPos, 1.0 - Math.exp(-dt * 2.8));
    this.camera.lookAt(this.focusLerp);
  }

  cycleSequence(step = 1) {
    this.sequenceIndex = (this.sequenceIndex + step + this.sequences.length) % this.sequences.length;
    this.sequenceTime = 0;
    this.onSequenceChange?.(this.sequenceIndex);
  }

  flyToPose(cameraState) {
    if (!cameraState?.position || !cameraState?.quaternion) return;
    const targetPos = cameraState.position.clone();
    const targetQuat = cameraState.quaternion.clone().normalize();
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(targetQuat).normalize();
    const focus = targetPos.clone().add(forward.multiplyScalar(this.sceneRadius * 0.8));
    const startPos = this.camera.position.clone();
    const startQuat = this.camera.quaternion.clone().normalize();
    const startFocus = this.focus.clone();
    const animState = { t: 0 };
    this.manualCameraState = {
      position: targetPos.clone(),
      quaternion: targetQuat.clone(),
    };
    this.isPoseTransitioning = true;
    gsap.killTweensOf(this.camera.position);
    gsap.killTweensOf(this.focus);
    gsap.killTweensOf(animState);
    gsap.to(animState, {
      duration: 1.2,
      t: 1,
      ease: 'power2.inOut',
      onUpdate: () => {
        this.camera.position.lerpVectors(startPos, targetPos, animState.t);
        this.camera.quaternion.slerpQuaternions(startQuat, targetQuat, animState.t).normalize();
        this.focus.lerpVectors(startFocus, focus, animState.t);
        this.onFocusChange?.(this.focus);
      },
      onComplete: () => {
        this.camera.position.copy(targetPos);
        this.camera.quaternion.copy(targetQuat);
        this.focus.copy(focus);
        this.focusLerp.copy(focus);
        this.isPoseTransitioning = false;
      },
    });
  }
}

export class PerformanceIO {
  constructor({ mount, onPayload, onUiChange, onSelectPose, onSequenceCycle, posesProvider }) {
    this.mount = mount;
    this.onPayload = onPayload;
    this.onUiChange = onUiChange;
    this.onSelectPose = onSelectPose;
    this.onSequenceCycle = onSequenceCycle;
    this.posesProvider = posesProvider;
    this.state = {
      progress: 0.22,
      fieldMode: 0,
      bloomStrength: 0.18,
      afterimageDamp: 0.82,
      pinch: 0,
      gestureEnabled: false,
      autoCamera: true,
      showConsole: window.innerWidth > 820,
      showHelp: false,
    };
    this.video = null;
    this.stream = null;
    this.handLandmarker = null;
    this.lastVideoTime = -1;
    this.bezier = [0.0, 0.18, 0.82, 1.0];
    this.handLoopId = 0;
    this.dom = {};
    this.currentPoseLabel = '等待载入场景';
    this.sceneSummary = '正在连接 Spark 渲染器';
    this.lastPayloadSignature = '';
    this._buildDom();
    this._bindFlutterBridge();
  }

  _buildDom() {
    this.mount.innerHTML = `
      <section class="bd-topbar">
        <div class="bd-brand">
          <p class="bd-brand-copy" data-role="summary">${this.sceneSummary}</p>
        </div>
        <div class="bd-toolbar">
          <div class="bd-stats">
            <div class="bd-chip" data-role="fps">帧率 --</div>
            <div class="bd-chip" data-role="gesture">手势 关闭</div>
          </div>
          <div class="bd-actions">
            <button class="bd-chip bd-chip--action" data-action="toggleConsole">控制台</button>
          </div>
        </div>
      </section>
      <section class="bd-focus-card" data-role="focusCard">
        <strong data-role="focusLabel">${this.currentPoseLabel}</strong>
        <span data-role="focusHint">点击下方镜头卡片可跳转到对应参考图</span>
      </section>
      <aside class="bd-console ${this.state.showConsole ? 'is-open' : ''}" data-role="console">
        <div class="bd-console__header">
          <div>
            <strong>控制台</strong>
          </div>
          <button class="bd-icon-btn" data-action="nextSequence">下一镜头</button>
        </div>
        <label class="bd-control">
          <span>矢量场模式</span>
          <select data-role="fieldMode">${FIELD_PRESETS.map((name, index) => `<option value="${index}">${index + 1}. ${name}</option>`).join('')}</select>
        </label>
        <label class="bd-control">
          <span>形变进度</span>
          <input type="range" min="0" max="1" step="0.001" value="${this.state.progress}" data-role="progress" />
        </label>
        <label class="bd-control">
          <span>泛光强度</span>
          <input type="range" min="0" max="2" step="0.01" value="${this.state.bloomStrength}" data-role="bloom" />
        </label>
        <label class="bd-control">
          <span>残影保留</span>
          <input type="range" min="0.72" max="0.95" step="0.001" value="${this.state.afterimageDamp}" data-role="afterimage" />
        </label>
        <label class="bd-control bd-control--toggle">
          <span>自动环绕运镜</span>
          <input type="checkbox" checked data-role="autoCamera" />
        </label>
        <label class="bd-control bd-control--toggle">
          <span>MediaPipe 捏合</span>
          <input type="checkbox" data-role="gesture" />
        </label>
        <div class="bd-bezier">
          <div class="bd-kicker">Bezier 映射</div>
          <input type="range" min="0" max="1" step="0.01" value="${this.bezier[1]}" data-role="bezier1" />
          <input type="range" min="0" max="1" step="0.01" value="${this.bezier[2]}" data-role="bezier2" />
        </div>
      </aside>
      <section class="bd-pose-dock">
        <aside class="bd-pose-rail" data-role="poseRail"></aside>
      </section>
      <div class="bd-gesture-preview" data-role="gesturePreview"></div>
    `;

    this.dom.fps = this.mount.querySelector('[data-role="fps"]');
    this.dom.gesture = this.mount.querySelector('[data-role="gesture"]');
    this.dom.console = this.mount.querySelector('[data-role="console"]');
    this.dom.helpPanel = this.mount.querySelector('[data-role="helpPanel"]');
    this.dom.summary = this.mount.querySelector('[data-role="summary"]');
    this.dom.focusLabel = this.mount.querySelector('[data-role="focusLabel"]');
    this.dom.focusHint = this.mount.querySelector('[data-role="focusHint"]');
    this.dom.fieldMode = this.mount.querySelector('[data-role="fieldMode"]');
    this.dom.progress = this.mount.querySelector('[data-role="progress"]');
    this.dom.bloom = this.mount.querySelector('[data-role="bloom"]');
    this.dom.afterimage = this.mount.querySelector('[data-role="afterimage"]');
    this.dom.autoCamera = this.mount.querySelector('[data-role="autoCamera"]');
    this.dom.gestureToggle = this.mount.querySelector('[data-role="gesture"]');
    this.dom.poseRail = this.mount.querySelector('[data-role="poseRail"]');
    this.dom.bezier1 = this.mount.querySelector('[data-role="bezier1"]');
    this.dom.bezier2 = this.mount.querySelector('[data-role="bezier2"]');
    this.dom.gesturePreview = this.mount.querySelector('[data-role="gesturePreview"]');

    this.mount.querySelector('[data-action="toggleConsole"]').addEventListener('click', () => {
      this.state.showConsole = !this.state.showConsole;
      this.dom.console.classList.toggle('is-open', this.state.showConsole);
    });
    this.mount.querySelector('[data-action="nextSequence"]').addEventListener('click', () => this.onSequenceCycle?.());

    this.dom.fieldMode.addEventListener('change', () => {
      this.state.fieldMode = Number(this.dom.fieldMode.value);
      this._emit();
    });
    this.dom.progress.addEventListener('input', () => {
      this.state.progress = Number(this.dom.progress.value);
      this._emit();
    });
    this.dom.bloom.addEventListener('input', () => {
      this.state.bloomStrength = Number(this.dom.bloom.value);
      this._emit();
    });
    this.dom.afterimage.addEventListener('input', () => {
      this.state.afterimageDamp = Number(this.dom.afterimage.value);
      this._emit();
    });
    this.dom.autoCamera.addEventListener('change', () => {
      this.state.autoCamera = this.dom.autoCamera.checked;
      this._emit();
    });
    this.dom.gestureToggle.addEventListener('change', async () => {
      this.state.gestureEnabled = this.dom.gestureToggle.checked;
      if (this.state.gestureEnabled) {
        await this.startGestureTracking();
      } else {
        this.stopGestureTracking();
      }
      this._emit();
    });
    this.dom.bezier1.addEventListener('input', () => {
      this.bezier[1] = Number(this.dom.bezier1.value);
    });
    this.dom.bezier2.addEventListener('input', () => {
      this.bezier[2] = Number(this.dom.bezier2.value);
    });
  }

  _bindFlutterBridge() {
    window.loadModelFromFlutter = (payload) => {
      const normalizedPayload = typeof payload === 'string'
        ? { ply: payload }
        : (payload && typeof payload === 'object' ? payload : null);
      if (!normalizedPayload) return;
      const signature = JSON.stringify({
        ply: normalizedPayload.ply || '',
        poses: normalizedPayload.poses || '',
        imageId: normalizedPayload.imageId || '',
        matrix: normalizedPayload.matrix || null,
      });
      if (signature === this.lastPayloadSignature) {
        return;
      }
      this.lastPayloadSignature = signature;
      if (typeof payload === 'string') {
        this.onPayload?.({ ply: payload });
        return;
      }
      if (payload && typeof payload === 'object') {
        this.onPayload?.(payload);
      }
    };
    if (window.BrainDanceChannel?.postMessage) {
      window.BrainDanceChannel.postMessage(JSON.stringify({ status: 'ready' }));
    }
  }

  _emit() {
    this.onUiChange?.({
      ...this.state,
      progressBezier: cubicBezier(0.0, this.bezier[1], this.bezier[2], 1.0, this.state.progress),
    });
  }

  setFps(value) {
    const rounded = Math.round(value);
    if (rounded === this._lastFps) return;
    this._lastFps = rounded;
    this.dom.fps.textContent = `帧率 ${rounded}`;
  }

  setGestureScalar(value) {
    const next = Math.round(value * 100) / 100;
    if (Math.abs((this._lastGestureScalar ?? -Infinity) - next) < 0.03) return;
    this._lastGestureScalar = next;
    this.state.pinch = value;
    this.dom.gesture.textContent = this.state.gestureEnabled ? `手势 ${next.toFixed(2)}` : '手势 关闭';
    this.dom.gesturePreview.style.setProperty('--gesture', `${next}`);
    this._emit();
  }

  setSequence(index) {
    this.mount.style.setProperty('--sequence-index', `${index}`);
  }

  bindPoses(poses) {
    const tagged = poses.filter((pose) => typeof pose?.tag === 'string' && pose.tag.trim().length > 0);
    const source = tagged.length > 0 ? tagged : [];
    const maxCards = window.innerWidth <= 520 ? 6 : 12;
    this.dom.poseRail.innerHTML = source.slice(0, maxCards).map((pose, index) => `
      <button class="bd-pose-card" data-pose-index="${index}">
        <strong>${pose.tag || pose.id || `镜头 ${index + 1}`}</strong>
        <span>${pose.image_url ? '参考图视角' : '标签镜头'}</span>
      </button>
    `).join('');
    this.dom.poseRail.querySelectorAll('[data-pose-index]').forEach((button) => {
      button.addEventListener('click', () => {
        const pose = source[Number(button.dataset.poseIndex)];
        if (pose) this.onSelectPose?.(pose);
      });
    });
  }

  setSceneSummary(summary) {
    this.sceneSummary = summary;
    if (this.dom.summary) {
      this.dom.summary.textContent = summary;
    }
  }

  setFocusPoseLabel(label, hint = '点击下方镜头卡片可跳转到对应参考图') {
    this.currentPoseLabel = label;
    if (this.dom.focusLabel) {
      this.dom.focusLabel.textContent = label;
    }
    if (this.dom.focusHint) {
      this.dom.focusHint.textContent = hint;
    }
  }

  async startGestureTracking() {
    if (this.handLandmarker) return;
    const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm');
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      },
      numHands: 1,
      runningMode: 'VIDEO',
    });
    this.video = document.createElement('video');
    this.video.autoplay = true;
    this.video.playsInline = true;
    this.video.muted = true;
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    this.video.srcObject = this.stream;

    const process = () => {
      this.handLoopId = requestAnimationFrame(process);
      if (!this.video || this.video.readyState < 2) return;
      if (this.video.currentTime === this.lastVideoTime) return;
      this.lastVideoTime = this.video.currentTime;
      const results = this.handLandmarker.detectForVideo(this.video, performance.now());
      const landmarks = results.landmarks?.[0];
      if (!landmarks) {
        this.setGestureScalar(0);
        return;
      }
      const wrist = landmarks[0];
      const thumb = landmarks[4];
      const index = landmarks[8];
      const middle = landmarks[9];
      const palm = Math.hypot(wrist.x - middle.x, wrist.y - middle.y) || 0.0001;
      const pinch = Math.hypot(thumb.x - index.x, thumb.y - index.y) / palm;
      const normalized = THREE.MathUtils.clamp(1.0 - pinch, 0.0, 1.0);
      this.setGestureScalar(normalized);
    };
    process();
  }

  stopGestureTracking() {
    cancelAnimationFrame(this.handLoopId);
    this.handLoopId = 0;
    this.stream?.getTracks()?.forEach((track) => track.stop());
    this.stream = null;
    this.video = null;
    this.handLandmarker?.close?.();
    this.handLandmarker = null;
    this.setGestureScalar(0);
  }

  dispose() {
    this.stopGestureTracking();
    delete window.loadModelFromFlutter;
  }
}
