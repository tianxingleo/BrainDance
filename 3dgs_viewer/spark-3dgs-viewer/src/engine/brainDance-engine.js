import * as THREE from 'three';
import { SparkControls, SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { AfterimagePass } from 'three/examples/jsm/postprocessing/AfterimagePass.js';
import { DataPipeline } from './data-pipeline.js';
import { CinematographyEngine, PerformanceIO } from './cinematography-io.js';
import { FIELD_PRESETS } from './vector-fields.js';

const DEFAULT_PLY = './models/scene_auto_sync_raw.ply';
const DEFAULT_POSES = './models/webgl_poses_with_tags.json';
const MOBILE_ROTATE_SPEED = 0.0055;
const MOBILE_PAN_SPEED = 0.0016;
const MOBILE_ZOOM_SPEED = 1.0;

function injectStyles() {
  const style = document.createElement('style');
  style.textContent = `
    :root {
      color-scheme: light;
      --bg-1: #f4f3ee;
      --bg-2: #e6e3db;
      --fg: #1e1e20;
      --muted: rgba(30, 30, 32, 0.66);
      --accent: #6b7a8f;
      --accent-2: #6d8260;
      --glass: rgba(249, 249, 248, 0.84);
      --glass-strong: rgba(249, 249, 248, 0.92);
      --border: rgba(107, 122, 143, 0.16);
      --shadow: 0 10px 26px rgba(0, 0, 0, 0.06);
      --gesture: 0;
      --sequence-index: 0;
      font-family: "HarmonyOS Sans SC", "Microsoft YaHei", "PingFang SC", sans-serif;
    }
    * { box-sizing: border-box; }
    html, body, #app {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      overscroll-behavior: none;
      background:
        radial-gradient(circle at top left, rgba(228, 232, 237, 0.16), transparent 24%),
        radial-gradient(circle at top right, rgba(107, 122, 143, 0.14), transparent 28%),
        linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
      color: var(--fg);
    }
    canvas { display: block; }
    .bd-root, .bd-canvas { position: absolute; inset: 0; }
    .bd-canvas,
    .bd-canvas canvas {
      touch-action: none;
      user-select: none;
      -webkit-user-select: none;
      -webkit-touch-callout: none;
    }
    .bd-ui {
      position: absolute;
      inset: 0;
      pointer-events: none;
    }
    .bd-topbar,
    .bd-console,
    .bd-pose-dock,
    .bd-pose-rail,
    .bd-gesture-preview,
    .bd-status { pointer-events: auto; }
    .bd-topbar {
      position: absolute;
      top: 18px;
      left: 18px;
      right: 18px;
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 12px;
      z-index: 22;
    }
    .bd-brand-copy {
      margin: 0;
      max-width: 420px;
      font-size: 13px;
      line-height: 1.5;
      color: var(--muted);
    }
    .bd-kicker {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: #6b7a8f;
    }
    .bd-toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      align-self: flex-end;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .bd-stats {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-start;
    }
    .bd-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .bd-chip, .bd-icon-btn, .bd-pose-card, select, input[type="range"] {
      border: 1px solid var(--border);
      background: var(--glass);
      color: var(--fg);
      backdrop-filter: blur(18px);
      box-shadow: var(--shadow);
    }
    button,
    input,
    select,
    textarea {
      touch-action: manipulation;
    }
    .bd-chip, .bd-icon-btn {
      padding: 10px 14px;
      border-radius: 14px;
      font: inherit;
      font-size: 13px;
      font-weight: 600;
    }
    .bd-chip--action, .bd-icon-btn, .bd-pose-card { cursor: pointer; }
    .bd-console {
      position: absolute;
      top: auto;
      right: 18px;
      left: auto;
      bottom: 108px;
      width: min(300px, calc(100vw - 36px));
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: var(--glass);
      backdrop-filter: blur(18px);
      box-shadow: var(--shadow);
      transform: translateX(calc(100% + 20px));
      transition: transform 260ms ease;
      display: grid;
      gap: 10px;
      touch-action: pan-y;
      z-index: 24;
    }
    .bd-focus-card {
      position: absolute;
      left: 18px;
      top: 116px;
      display: grid;
      gap: 6px;
      width: min(44vw, 320px);
      padding: 14px 16px;
      border-radius: 22px;
      border: 1px solid var(--border);
      background: var(--glass);
      backdrop-filter: blur(18px);
      box-shadow: var(--shadow);
      z-index: 21;
    }
    .bd-status {
      position: absolute;
      left: 18px;
      right: 18px;
      top: 96px;
      padding: 10px 14px;
      border-radius: 18px;
      border: 1px solid rgba(139, 71, 71, 0.18);
      background: rgba(249, 249, 248, 0.92);
      color: #8b4747;
      font-size: 12px;
      line-height: 1.5;
      opacity: 0;
      transform: translateY(-8px);
      transition: opacity 180ms ease, transform 180ms ease;
      z-index: 30;
      white-space: pre-wrap;
    }
    .bd-status.is-visible {
      opacity: 1;
      transform: translateY(0);
    }
    .bd-focus-card strong {
      font-size: 15px;
      font-weight: 700;
      line-height: 1.4;
    }
    .bd-focus-card span {
      font-size: 12px;
      line-height: 1.5;
      color: var(--muted);
    }
    .bd-console.is-open { transform: translateX(0); }
    .bd-console__header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .bd-control {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }
    .bd-control--toggle {
      grid-template-columns: 1fr auto;
      align-items: center;
    }
    .bd-control select, .bd-control input[type="range"] {
      width: 100%;
      border-radius: 14px;
      padding: 10px 12px;
      accent-color: #6d8260;
      box-shadow: none;
    }
    .bd-bezier {
      display: grid;
      gap: 8px;
      padding: 12px;
      border-radius: 18px;
      background: rgba(107, 122, 143, 0.08);
    }
    .bd-pose-dock {
      position: absolute;
      left: 18px;
      right: 18px;
      bottom: max(18px, env(safe-area-inset-bottom, 0px));
      width: auto;
      display: grid;
      gap: 0;
      z-index: 23;
    }
    .bd-pose-rail {
      position: relative;
      display: flex;
      align-items: stretch;
      gap: 16px;
      overflow-x: auto;
      padding: 16px 18px;
      background: var(--glass);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      touch-action: pan-x;
      scroll-snap-type: x proximity;
    }
    .bd-pose-card {
      flex: 0 0 auto;
      width: 100px;
      min-width: 100px;
      min-height: 70px;
      border-radius: 16px;
      padding: 10px 12px;
      text-align: left;
      display: grid;
      align-content: end;
      gap: 4px;
      scroll-snap-align: start;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(107, 122, 143, 0.12);
      box-shadow: 0 6px 12px rgba(0, 0, 0, 0.04);
      transition: all 0.25s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .bd-pose-card strong {
      font-size: 12px;
      font-weight: 700;
    }
    .bd-pose-card span {
      font-size: 11px;
      color: var(--muted);
    }
    .bd-pose-card:hover,
    .bd-pose-card:focus-visible {
      transform: translateY(-3px);
      box-shadow: 0 10px 20px rgba(107, 122, 143, 0.12);
      outline: none;
    }
    .bd-gesture-preview {
      position: absolute;
      left: 18px;
      top: 112px;
      width: 140px;
      height: 140px;
      border-radius: 50%;
      border: 1px solid rgba(107, 122, 143, 0.18);
      background:
        radial-gradient(circle, rgba(109,130,96, calc(0.12 + var(--gesture) * 0.28)), transparent 52%),
        conic-gradient(from calc(var(--sequence-index) * 24deg), rgba(107,122,143,0.28), rgba(107,122,143,0.04), rgba(107,122,143,0.28));
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.06);
      pointer-events: none;
    }
    @media (max-width: 820px) {
      .bd-topbar {
        top: 18px;
        left: 18px;
        right: 18px;
      }
      .bd-brand-copy { max-width: none; }
      .bd-toolbar {
        align-self: flex-start;
      }
      .bd-stats,
      .bd-actions {
        justify-content: flex-start;
      }
      .bd-focus-card {
        display: grid;
        left: 18px;
        top: 108px;
        width: min(60vw, 320px);
      }
      .bd-status {
        left: 12px;
        right: 12px;
        top: 88px;
      }
      .bd-console {
        bottom: 116px;
        right: 12px;
        left: 12px;
        width: auto;
        border-radius: 24px;
      }
      .bd-pose-dock {
        left: 12px;
        right: 12px;
        width: auto;
        bottom: max(10px, env(safe-area-inset-bottom, 0px));
      }
      .bd-gesture-preview { width: 88px; height: 88px; top: 244px; left: auto; right: 12px; }
    }
    @media (orientation: portrait) and (max-width: 820px) {
      .bd-topbar {
        top: 12px;
        left: 12px;
        right: 12px;
        gap: 10px;
      }
      .bd-brand-copy {
        font-size: 12px;
        max-width: none;
      }
      .bd-chip {
        padding: 8px 11px;
        font-size: 12px;
      }
      .bd-pose-dock {
        bottom: max(8px, env(safe-area-inset-bottom, 0px));
      }
      .bd-console {
        max-height: 33vh;
        overflow: auto;
        padding: 12px;
        z-index: 20;
        border-radius: 20px;
        bottom: 96px;
      }
      .bd-gesture-preview {
        display: none;
      }
      .bd-focus-card {
        top: 96px;
        z-index: 16;
        width: min(72vw, 280px);
      }
      .bd-pose-dock {
        z-index: 18;
      }
      .bd-pose-rail {
        display: grid;
        grid-auto-flow: column;
        grid-auto-columns: 100px;
        padding: 14px;
        gap: 12px;
      }
    }
    @media (orientation: portrait) and (max-width: 480px) {
      .bd-console {
        max-height: 31vh;
      }
      .bd-pose-dock {
        bottom: max(8px, env(safe-area-inset-bottom, 0px));
      }
      .bd-pose-rail {
        grid-auto-columns: minmax(104px, 40vw);
      }
      .bd-status {
        top: 82px;
      }
      .bd-actions {
        width: 100%;
      }
      .bd-actions .bd-chip {
        flex: 1 1 auto;
        text-align: center;
      }
      .bd-focus-card {
        top: 88px;
        padding: 10px 12px;
      }
    }
  `;
  document.head.appendChild(style);
}

function sendChannelMessage(payload) {
  try {
    window.BrainDanceChannel?.postMessage?.(JSON.stringify(payload));
  } catch (_) {}
}

function normalizeImageId(value) {
  if (value == null) return '';
  const text = String(value).trim().replace(/\\/g, '/');
  return (text.split('/').pop() || '').toLowerCase();
}

function parseUrlPayload() {
  const params = new URLSearchParams(window.location.search);
  const payload = params.get('payload');
  if (payload) {
    try {
      return JSON.parse(decodeURIComponent(payload));
    } catch (error) {
      console.warn('[BrainDance] Failed to parse payload.', error);
    }
  }
  const matrixParam = params.get('matrix');
  let matrix = null;
  if (matrixParam) {
    try {
      matrix = JSON.parse(decodeURIComponent(matrixParam));
    } catch (error) {
      console.warn('[BrainDance] Failed to parse matrix.', error);
    }
  }
  return {
    ply: params.get('ply') || null,
    poses: params.get('poses') || null,
    imageId: params.get('imageId') || null,
    matrix,
  };
}

function getPoseImageId(pose) {
  return normalizeImageId(pose.id || pose.image_id || pose.imageId || pose.image_url || '');
}

function normalizeMatrixValues(matrixLike) {
  const values = Array.isArray(matrixLike?.[0]) ? matrixLike.flat() : matrixLike;
  if (!Array.isArray(values) || values.length !== 16) return null;
  const normalized = values.map((value) => Number(value));
  return normalized.every((value) => Number.isFinite(value)) ? normalized : null;
}

class BrainDanceEngine {
  constructor(mount) {
    this.mount = mount;
    this.clock = new THREE.Clock();
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x03050d, 0.08);
    this.root = document.createElement('div');
    this.root.className = 'bd-root';
    this.canvasLayer = document.createElement('div');
    this.canvasLayer.className = 'bd-canvas';
    this.uiLayer = document.createElement('div');
    this.uiLayer.className = 'bd-ui';
    this.root.append(this.canvasLayer, this.uiLayer);
    this.mount.appendChild(this.root);
    this.statusEl = document.createElement('div');
    this.statusEl.className = 'bd-status';
    this.root.appendChild(this.statusEl);
    this.isMobile =
      window.matchMedia('(pointer: coarse)').matches ||
      'ontouchstart' in window ||
      (navigator.maxTouchPoints || 0) > 0 ||
      window.innerWidth <= 820;
    this.touchState = {
      dragging: false,
      pinching: false,
      lastX: 0,
      lastY: 0,
      lastDistance: 0,
      lastMidX: 0,
      lastMidY: 0,
    };
    this.uiInteracting = false;

    this.renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.setClearColor(0x040816, 1);
    this.canvasLayer.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.touchAction = 'none';
    this.renderer.domElement.style.userSelect = 'none';
    this.renderer.domElement.style.webkitUserSelect = 'none';
    ['touchstart', 'touchmove', 'gesturestart', 'gesturechange', 'gestureend'].forEach((eventName) => {
      this.renderer.domElement.addEventListener(eventName, (event) => {
        event.preventDefault();
      }, { passive: false });
    });

    this.camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.01, 64);
    this.camera.position.set(0, 0.24, 3.1);
    this.scene.add(this.camera);

    this.spark = new SparkRenderer({
      renderer: this.renderer,
      maxStdDev: Math.sqrt(7),
      preUpdate: false,
      view: { sortRadial: true },
    });
    this.scene.add(this.spark);

    this.controls = new SparkControls({ canvas: this.renderer.domElement });
    this.controls.fpsMovement.enable = false;
    this.controls.pointerControls.rotateSpeed = window.innerWidth <= 820 ? 0.0022 : 0.0015;
    this.controls.pointerControls.slideSpeed = window.innerWidth <= 820 ? 0.0062 : 0.0042;
    this.controls.pointerControls.scrollSpeed = window.innerWidth <= 820 ? 0.0010 : 0.0014;
    this.controls.pointerControls.reverseSwipe = true;
    this.controls.pointerControls.moveInertia = 0.82;
    this.controls.pointerControls.rotateInertia = 0.78;
    if (this.isMobile) {
      this.controls.pointerControls.enable = false;
    }

    this.composer = new EffectComposer(this.renderer);
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.42, 0.45, 0.32);
    this.afterimagePass = new AfterimagePass();
    this.afterimagePass.uniforms.damp.value = 0.82;
    this.composer.addPass(this.renderPass);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(this.afterimagePass);
    this.usePostProcessing = true;

    this.pipeline = new DataPipeline({ renderer: this.renderer });
    this.splatMesh = null;
    this.poses = [];
    this.sceneCenter = new THREE.Vector3();
    this.sceneRadius = 1.1;
    this.runtime = {
      fieldMode: 0,
      progress: 0.22,
      progressBezier: 0.22,
      pinch: 0,
      bloomStrength: 0.42,
      afterimageDamp: 0.82,
      autoCamera: true,
    };
    this._isAnimating = false;
    this._loadToken = 0;
    this._activeLoadSignature = '';

    this.cinema = new CinematographyEngine({
      camera: this.camera,
      controls: this.controls,
      sceneRadius: this.sceneRadius,
      sceneCenter: this.sceneCenter,
      poses: this.poses,
      onFocusChange: (focus) => {
        this.focusPoint = focus.clone();
      },
      onSequenceChange: (index) => this.io?.setSequence(index),
    });
    this.cinema.frameFocus(this.sceneCenter);

    this.io = new PerformanceIO({
      mount: this.uiLayer,
      onPayload: (payload) => this.loadScene(payload),
      onUiChange: (state) => {
        Object.assign(this.runtime, state);
        this.cinema.autoPlay = state.autoCamera;
        this._applyRuntimeVisuals();
      },
      onSelectPose: (pose) => this.focusPose(pose),
      onSequenceCycle: () => this.cinema.cycleSequence(1),
      posesProvider: () => this.poses,
    });

    this._addLighting();
    this._installGlobalErrorHooks();
    this._installMobileTouchControls();
    this._installUiGestureGuards();
    window.addEventListener('resize', () => this.onResize());
  }

  _setUiInteraction(active) {
    this.uiInteracting = active;
    this.touchState.dragging = false;
    this.touchState.pinching = false;
    this.touchState.lastDistance = 0;
    if (this.canvasLayer) {
      this.canvasLayer.style.pointerEvents = active ? 'none' : 'auto';
    }
    if (active && this.controls?.pointerControls) {
      this.controls.pointerControls.enable = false;
    } else if (this.controls?.pointerControls && !this.isMobile) {
      this.controls.pointerControls.enable = true;
    }
  }

  _applyRuntimeVisuals() {
    this.bloomPass.strength = this.runtime.bloomStrength;
    this.afterimagePass.uniforms.damp.value = this.runtime.afterimageDamp;

    if (this.pipeline?.renderMaterial?.uniforms?.uVisibility) {
      const visibility = this.splatMesh
        ? THREE.MathUtils.lerp(0.04, 0.34, this.runtime.progressBezier)
        : THREE.MathUtils.lerp(0.12, 0.42, this.runtime.progressBezier);
      this.pipeline.renderMaterial.uniforms.uVisibility.value = visibility;
    }

    if (this.pipeline?.points) {
      this.pipeline.points.visible = true;
      const scale = 1 + this.runtime.progressBezier * 0.04;
      this.pipeline.points.scale.setScalar(scale);
    }

    this.usePostProcessing = true;
  }

  _installUiGestureGuards() {
    const startUiInteraction = () => {
      this._setUiInteraction(true);
    };
    const endUiInteraction = () => {
      this._setUiInteraction(false);
    };
    const stop = (event) => {
      event.stopPropagation();
      if (typeof event.stopImmediatePropagation === 'function') {
        event.stopImmediatePropagation();
      }
      if (event.type.startsWith('touch') && event.touches && event.touches.length > 1) {
        event.preventDefault();
      }
    };
    const stopSelectors = ['button', 'input', 'select', '.bd-console', '.bd-help', '.bd-pose-dock', '.bd-status', '.bd-topbar', '.bd-focus-card'];
    stopSelectors.forEach((selector) => {
      this.root.querySelectorAll(selector).forEach((node) => {
        ['touchstart', 'touchmove', 'touchend', 'touchcancel', 'pointerdown', 'pointermove', 'pointerup', 'pointercancel', 'mousedown', 'mouseup', 'wheel'].forEach((eventName) => {
          const isTouch = eventName.startsWith('touch');
          node.addEventListener(eventName, stop, { passive: !isTouch, capture: true });
        });
        ['touchstart', 'pointerdown', 'mousedown', 'focusin'].forEach((eventName) => {
          node.addEventListener(eventName, startUiInteraction, { passive: true, capture: true });
        });
        ['touchend', 'touchcancel', 'pointerup', 'pointercancel', 'mouseup', 'focusout', 'change'].forEach((eventName) => {
          node.addEventListener(eventName, endUiInteraction, { passive: true, capture: true });
        });
      });
    });
    ['pointerup', 'pointercancel', 'mouseup', 'touchend', 'touchcancel'].forEach((eventName) => {
      window.addEventListener(eventName, endUiInteraction, { passive: true, capture: true });
    });
  }

  _installMobileTouchControls() {
    const surface = this.root;
    const getDistance = (a, b) => Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    const getMid = (a, b) => ({
      x: (a.clientX + b.clientX) * 0.5,
      y: (a.clientY + b.clientY) * 0.5,
    });
    const isUiTarget = (target) => target instanceof Element && Boolean(target.closest('.bd-topbar, .bd-console, .bd-help, .bd-pose-dock, .bd-focus-card, .bd-status'));

    const onTouchStart = (event) => {
      if (!this.isMobile) return;
      if (this.uiInteracting) return;
      if (isUiTarget(event.target)) return;
      if (event.touches.length >= 2) {
        const [a, b] = event.touches;
        const mid = getMid(a, b);
        this.touchState.dragging = false;
        this.touchState.pinching = true;
        this.touchState.lastDistance = getDistance(a, b);
        this.touchState.lastMidX = mid.x;
        this.touchState.lastMidY = mid.y;
        return;
      }
      if (event.touches.length === 1) {
        this.touchState.dragging = true;
        this.touchState.pinching = false;
        this.touchState.lastX = event.touches[0].clientX;
        this.touchState.lastY = event.touches[0].clientY;
      }
    };

    const onTouchMove = (event) => {
      if (!this.isMobile) return;
      if (!this.camera) return;
      if (this.uiInteracting) return;
      if (isUiTarget(event.target)) return;
      const focus = this.cinema.focus.clone();

      if (event.touches.length >= 2) {
        event.preventDefault();
        const [a, b] = event.touches;
        const nextDistance = getDistance(a, b);
        const mid = getMid(a, b);

        if (this.touchState.pinching && this.touchState.lastDistance > 0) {
          const scale = nextDistance / this.touchState.lastDistance;
          this._mobileZoom(1 + ((scale - 1) * MOBILE_ZOOM_SPEED), focus);
        }

        const panDx = mid.x - this.touchState.lastMidX;
        const panDy = mid.y - this.touchState.lastMidY;
        this._mobilePan(panDx, panDy);

        this.touchState.pinching = true;
        this.touchState.lastDistance = nextDistance;
        this.touchState.lastMidX = mid.x;
        this.touchState.lastMidY = mid.y;
        this.touchState.dragging = false;
        return;
      }

      if (event.touches.length === 1 && this.touchState.dragging) {
        event.preventDefault();
        const touch = event.touches[0];
        const dx = touch.clientX - this.touchState.lastX;
        const dy = touch.clientY - this.touchState.lastY;
        this._mobileOrbit(dx, dy, focus);
        this.touchState.lastX = touch.clientX;
        this.touchState.lastY = touch.clientY;
      }
    };

    const onTouchEnd = (event) => {
      if (!this.isMobile) return;
      if (this.uiInteracting && event.touches.length === 0) {
        this.touchState.dragging = false;
        this.touchState.pinching = false;
        this.touchState.lastDistance = 0;
        return;
      }
      if (event.touches.length >= 2) {
        const [a, b] = event.touches;
        const mid = getMid(a, b);
        this.touchState.pinching = true;
        this.touchState.lastDistance = getDistance(a, b);
        this.touchState.lastMidX = mid.x;
        this.touchState.lastMidY = mid.y;
        this.touchState.dragging = false;
        return;
      }
      if (event.touches.length === 1) {
        this.touchState.dragging = true;
        this.touchState.pinching = false;
        this.touchState.lastX = event.touches[0].clientX;
        this.touchState.lastY = event.touches[0].clientY;
        return;
      }
      this.touchState.dragging = false;
      this.touchState.pinching = false;
      this.touchState.lastDistance = 0;
    };

    surface.addEventListener('touchstart', onTouchStart, { passive: false });
    surface.addEventListener('touchmove', onTouchMove, { passive: false });
    surface.addEventListener('touchend', onTouchEnd, { passive: false });
    surface.addEventListener('touchcancel', onTouchEnd, { passive: false });
  }

  _mobileOrbit(dx, dy, focus) {
    this.cinema.manualCameraState = null;
    const offset = this.camera.position.clone().sub(focus);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    spherical.theta -= dx * MOBILE_ROTATE_SPEED;
    spherical.phi += dy * MOBILE_ROTATE_SPEED;
    spherical.phi = THREE.MathUtils.clamp(spherical.phi, 0.18, Math.PI - 0.18);
    offset.setFromSpherical(spherical);
    this.camera.position.copy(focus.clone().add(offset));
    this.camera.lookAt(focus);
    this.cinema.autoPlay = false;
    this.runtime.autoCamera = false;
    if (this.io?.dom?.autoCamera) {
      this.io.dom.autoCamera.checked = false;
    }
  }

  _mobilePan(dx, dy) {
    this.cinema.manualCameraState = null;
    const distance = Math.max(0.25, this.camera.position.distanceTo(this.cinema.focus));
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    const offset = new THREE.Vector3()
      .addScaledVector(right, -dx * MOBILE_PAN_SPEED * distance)
      .addScaledVector(up, dy * MOBILE_PAN_SPEED * distance);
    this.camera.position.add(offset);
    this.cinema.focus.add(offset);
    this.cinema.focusLerp.add(offset);
    this.camera.lookAt(this.cinema.focus);
    this.cinema.autoPlay = false;
    this.runtime.autoCamera = false;
    if (this.io?.dom?.autoCamera) {
      this.io.dom.autoCamera.checked = false;
    }
  }

  _mobileZoom(scale, focus) {
    this.cinema.manualCameraState = null;
    if (!Number.isFinite(scale) || scale <= 0) return;
    const offset = this.camera.position.clone().sub(focus);
    const distance = THREE.MathUtils.clamp(offset.length() / Math.max(scale, 0.25), this.sceneRadius * 0.6, this.sceneRadius * 6.0);
    offset.setLength(distance);
    this.camera.position.copy(focus.clone().add(offset));
    this.camera.lookAt(focus);
    this.cinema.autoPlay = false;
    this.runtime.autoCamera = false;
    if (this.io?.dom?.autoCamera) {
      this.io.dom.autoCamera.checked = false;
    }
  }

  _installGlobalErrorHooks() {
    window.addEventListener('error', (event) => {
      const message = event?.error?.stack || event?.message || '未知脚本错误';
      this.setStatus(`前端脚本错误\n${message}`, true);
    });
    window.addEventListener('unhandledrejection', (event) => {
      const reason = event?.reason?.stack || event?.reason?.message || String(event?.reason || 'Promise rejected');
      this.setStatus(`异步任务失败\n${reason}`, true);
    });
  }

  setStatus(message, isError = false) {
    if (!this.statusEl) return;
    this.statusEl.textContent = message;
    this.statusEl.classList.toggle('is-visible', Boolean(message));
    this.statusEl.style.borderColor = isError ? 'rgba(255, 155, 113, 0.44)' : 'rgba(135, 255, 225, 0.24)';
    this.statusEl.style.background = isError ? 'rgba(28, 12, 10, 0.82)' : 'rgba(7, 18, 32, 0.72)';
    this.statusEl.style.color = isError ? '#ffd9c8' : '#d5fff6';
    sendChannelMessage({
      status: isError ? 'error' : 'info',
      msg: message,
    });
  }

  clearStatus() {
    if (!this.statusEl) return;
    this.statusEl.textContent = '';
    this.statusEl.classList.remove('is-visible');
  }

  _addLighting() {
    const hemi = new THREE.HemisphereLight(0xb8fff0, 0x120813, 0.55);
    const dir = new THREE.DirectionalLight(0xffd8c2, 1.4);
    dir.position.set(2.4, 3.0, 1.8);
    this.scene.add(hemi, dir);
  }

  async init() {
    if (window.BrainDanceChannel) {
      this.setStatus('等待 Flutter 发送模型数据');
      return;
    }
    const payload = parseUrlPayload();
    await this.loadScene(payload);
  }

  async loadScene(payload = {}) {
    const loadSignature = JSON.stringify({
      ply: payload?.ply || DEFAULT_PLY,
      poses: payload?.poses || DEFAULT_POSES,
      imageId: payload?.imageId || '',
      matrix: payload?.matrix || null,
    });
    if (loadSignature === this._activeLoadSignature) {
      return;
    }
    this._activeLoadSignature = loadSignature;
    const loadToken = ++this._loadToken;
    const ply = payload?.ply || DEFAULT_PLY;
    const posesUrl = payload?.poses || DEFAULT_POSES;
    const isSplatLike = String(ply || '').toLowerCase().endsWith('.splat') || String(ply || '').toLowerCase().endsWith('.ksplat');
    this.setStatus(`准备加载模型\n${ply}`);

    if (this.splatMesh) {
      this.splatMesh.removeFromParent();
      this.splatMesh.dispose();
      this.splatMesh = null;
    }
    if (this.pipeline.points) {
      this.scene.remove(this.pipeline.points);
    }
    this.pipeline.dispose();
    this.pipeline = new DataPipeline({ renderer: this.renderer });
    await this.pipeline.load(ply);
    if (loadToken !== this._loadToken) return;
    if (this.pipeline.points) {
      this.scene.add(this.pipeline.points);
    }
    if (this.pipeline.renderMaterial?.uniforms?.uVisibility) {
      this.pipeline.renderMaterial.uniforms.uVisibility.value = isSplatLike ? 0.04 : 0.14;
    }
    if (this.pipeline.points) {
      this.pipeline.points.visible = true;
    }
    this.usePostProcessing = true;
    this.setStatus(`PLY 请求完成\n${ply}`);

    try {
      this.splatMesh = new SplatMesh({ url: ply, editable: true });
      this.scene.add(this.splatMesh);
      this.setStatus('Spark 正在初始化高斯模型');
      await this.splatMesh.initialized;
      if (loadToken !== this._loadToken) return;
      const bbox = this.splatMesh.getBoundingBox(true);
      const size = bbox.getSize(new THREE.Vector3());
      this.sceneCenter = bbox.getCenter(new THREE.Vector3());
      this.sceneRadius = Math.max(size.length() * 0.24, this.pipeline.bounds.radius, 0.8);
      this.io.setSceneSummary('已载入真实 3DGS 场景。默认以原始高斯渲染为主，粒子矢量场作为辅助特效叠层。');
      this._applyRuntimeVisuals();
      this.clearStatus();
    } catch (error) {
      console.warn('[BrainDance] Spark splat load failed, particles remain active.', error);
      this.sceneCenter.set(0, 0, 0);
      this.sceneRadius = this.pipeline.bounds.radius;
      this.io.setSceneSummary('未找到可用的 3DGS 模型文件，当前展示的是内置粒子演示场。');
      if (this.pipeline.renderMaterial?.uniforms?.uVisibility) {
        this.pipeline.renderMaterial.uniforms.uVisibility.value = 0.18;
      }
      if (this.pipeline.points) {
        this.pipeline.points.visible = true;
      }
      this.usePostProcessing = true;
      this._applyRuntimeVisuals();
      this.setStatus(`Spark 模型加载失败\n${error?.message || error}`, true);
    }

    this.cinema.setSceneBounds({
      radius: this.sceneRadius,
      center: this.sceneCenter,
    });
    this.cinema.frameFocus(this.sceneCenter);
    this.io.setFocusPoseLabel('场景总览', this.splatMesh ? '当前先展示整体高斯场景，点击下方镜头卡片查看参考视角。' : '当前不是白屏，而是内置粒子演示场。接入真实 PLY 后会切换为模型视图。');

    await this.loadPoses(posesUrl);
    if (loadToken !== this._loadToken) return;
    if (payload?.imageId) {
      const targetPose = this.poses.find((pose) => getPoseImageId(pose) === normalizeImageId(payload.imageId));
      if (targetPose) this.focusPose(targetPose);
    } else if (payload?.matrix) {
      this.focusPose({ matrix: payload.matrix });
    }

    sendChannelMessage({
      status: 'info',
      msg: 'Spark 场景已就绪',
    });
    if (!this._isAnimating) {
      this._isAnimating = true;
      this.animate();
    }
  }

  async loadPoses(url) {
    this.poses = [];
    if (!url) {
      this.io.bindPoses(this.poses);
      return;
    }
    try {
      const response = await fetch(url);
      const data = await response.json();
      this.poses = Array.isArray(data?.frames)
        ? data.frames.map((frame) => ({
            ...frame,
            tag: typeof frame.tag === 'string' ? frame.tag.trim() : '',
          }))
        : (Array.isArray(data) ? data : []);
    } catch (error) {
      console.warn('[BrainDance] Pose loading failed.', error);
      this.setStatus(`位姿文件加载失败\n${url}\n${error?.message || error}`, true);
    }
    this.io.bindPoses(this.poses);
  }

  resolvePoseCameraState(pose) {
    const matrixValues = normalizeMatrixValues(pose?.matrix);
    if (!matrixValues) return null;
    const rawMatrix = new THREE.Matrix4().fromArray(matrixValues);
    const finalMatrix = new THREE.Matrix4();
    if (this.splatMesh) {
      this.splatMesh.updateMatrixWorld(true);
      finalMatrix.copy(this.splatMesh.matrixWorld).multiply(rawMatrix);
    } else {
      finalMatrix.copy(rawMatrix);
    }

    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    finalMatrix.decompose(position, quaternion, scale);

    return {
      position,
      quaternion: quaternion.normalize(),
      fl_y: Number(pose?.fl_y || 0),
      h: Number(pose?.h || 0),
    };
  }

  focusPose(pose) {
    const cameraState = this.resolvePoseCameraState(pose);
    if (!cameraState) return;
    this.cinema.autoPlay = false;
    this.runtime.autoCamera = false;
    this.io.dom.autoCamera.checked = false;
    this.io.setFocusPoseLabel(
      pose.tag || pose.id || '已定位镜头',
      pose.image_url ? '当前镜头对应一张参考图片，运镜会朝这张图的采集视角靠近。' : '当前镜头对应一组相机位姿矩阵。'
    );
    this.cinema.flyToPose(cameraState);
  }

  animate() {
    if (!this._isAnimating) return;
    requestAnimationFrame(() => this.animate());
    const dt = Math.min(this.clock.getDelta(), 0.033);
    const elapsed = this.clock.elapsedTime;

    if (!this.isMobile && !this.uiInteracting) {
      this.controls.update(this.camera);
    }
    this.cinema.autoPlay = this.runtime.autoCamera;
    this.cinema.update(dt);

    this.pipeline.step({
      dt,
      time: elapsed,
      progress: this.runtime.progressBezier,
      pinch: this.runtime.pinch,
      fieldMode: this.runtime.fieldMode,
      viewportHeight: window.innerHeight,
    });

    this._applyRuntimeVisuals();

    if (this.usePostProcessing) {
      this.composer.render();
    } else {
      this.renderer.render(this.scene, this.camera);
    }
    this.io.setFps(1 / Math.max(dt, 1e-5));
  }

  onResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.composer.setSize(width, height);
    this.io.bindPoses(this.poses);
  }
}

injectStyles();
const app = document.getElementById('app');
const engine = new BrainDanceEngine(app);
engine.init();
