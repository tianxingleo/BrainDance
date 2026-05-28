import type {
  HandLandmarker,
  HandLandmarkerResult,
  NormalizedLandmark,
} from '@mediapipe/tasks-vision';

export type AirGestureAction =
  | { type: 'ready' }
  | { type: 'hand_present'; openness: number }
  | { type: 'open'; amount: number }
  | { type: 'close'; amount: number }
  | { type: 'swipe_left'; strength: number }
  | { type: 'swipe_right'; strength: number }
  | { type: 'lost_hand' }
  | { type: 'error'; message: string };

export type AirGestureStatus =
  | 'idle'
  | 'requesting_camera'
  | 'loading_model'
  | 'running'
  | 'hand_present'
  | 'lost_hand'
  | 'error';

export interface AirGestureControllerOptions {
  video: HTMLVideoElement;
  onAction: (action: AirGestureAction) => void;
  onStatus?: (status: AirGestureStatus, message?: string) => void;
  debug?: boolean;
  inferenceIntervalMs?: number;
  swipeCooldownMs?: number;
  pinchCooldownMs?: number;
  mirrorInput?: boolean;
  wasmBaseUrl?: string;
  modelAssetPath?: string;
}

type PalmSample = {
  timeMs: number;
  x: number;
  y: number;
  openness: number;
};

const isLandmark = (value: NormalizedLandmark | undefined): value is NormalizedLandmark => Boolean(value);

const DEFAULT_WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const DEFAULT_MODEL_ASSET_PATH =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const LANDMARK = {
  wrist: 0,
  indexMcp: 5,
  middleMcp: 9,
  ringMcp: 13,
  pinkyMcp: 17,
  indexTip: 8,
  middleTip: 12,
  ringTip: 16,
  pinkyTip: 20,
};

const distance2d = (a: NormalizedLandmark, b: NormalizedLandmark) => {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
};

const getLandmark = (landmarks: NormalizedLandmark[], index: number) => landmarks[index];

const computeOpenness = (landmarks: NormalizedLandmark[]) => {
  const wrist = getLandmark(landmarks, LANDMARK.wrist);
  const indexMcp = getLandmark(landmarks, LANDMARK.indexMcp);
  const pinkyMcp = getLandmark(landmarks, LANDMARK.pinkyMcp);
  const tips = [
    getLandmark(landmarks, LANDMARK.indexTip),
    getLandmark(landmarks, LANDMARK.middleTip),
    getLandmark(landmarks, LANDMARK.ringTip),
    getLandmark(landmarks, LANDMARK.pinkyTip),
  ].filter(isLandmark);
  if (!wrist || !indexMcp || !pinkyMcp || tips.length < 4) return 0;

  const palmWidth = Math.max(distance2d(indexMcp, pinkyMcp), 0.0001);
  const fingerReach = tips.reduce((sum, tip) => sum + distance2d(tip, wrist), 0) / tips.length;
  return fingerReach / palmWidth;
};

const computePalmVisualX = (landmarks: NormalizedLandmark[], mirrorInput: boolean) => {
  const points = [
    getLandmark(landmarks, LANDMARK.wrist),
    getLandmark(landmarks, LANDMARK.indexMcp),
    getLandmark(landmarks, LANDMARK.middleMcp),
    getLandmark(landmarks, LANDMARK.ringMcp),
    getLandmark(landmarks, LANDMARK.pinkyMcp),
  ].filter(isLandmark);
  if (!points.length) return 0.5;

  const rawX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
  return mirrorInput ? 1 - rawX : rawX;
};

const computePalmY = (landmarks: NormalizedLandmark[]) => {
  const points = [
    getLandmark(landmarks, LANDMARK.wrist),
    getLandmark(landmarks, LANDMARK.indexMcp),
    getLandmark(landmarks, LANDMARK.middleMcp),
    getLandmark(landmarks, LANDMARK.ringMcp),
    getLandmark(landmarks, LANDMARK.pinkyMcp),
  ].filter(isLandmark);
  if (!points.length) return 0.5;

  return points.reduce((sum, point) => sum + point.y, 0) / points.length;
};

export class AirGestureController {
  private readonly video: HTMLVideoElement;
  private readonly onAction: (action: AirGestureAction) => void;
  private readonly onStatus?: (status: AirGestureStatus, message?: string) => void;
  private readonly inferenceIntervalMs: number;
  private readonly swipeCooldownMs: number;
  private readonly pinchCooldownMs: number;
  private readonly mirrorInput: boolean;
  private readonly wasmBaseUrl: string;
  private readonly modelAssetPath: string;
  private readonly debug: boolean;

  private stream: MediaStream | null = null;
  private handLandmarker: HandLandmarker | null = null;
  private animationFrameId = 0;
  private lastInferenceMs = 0;
  private lastVideoTime = -1;
  private lastSwipeMs = 0;
  private lastPinchMs = 0;
  private hasHand = false;
  private palmSamples: PalmSample[] = [];
  private running = false;
  private lastStatus: AirGestureStatus | null = null;
  private lastDebugMs = 0;

  constructor(options: AirGestureControllerOptions) {
    this.video = options.video;
    this.onAction = options.onAction;
    this.onStatus = options.onStatus;
    this.debug = options.debug ?? true;
    this.inferenceIntervalMs = options.inferenceIntervalMs ?? 100;
    this.swipeCooldownMs = options.swipeCooldownMs ?? 520;
    this.pinchCooldownMs = options.pinchCooldownMs ?? 260;
    this.mirrorInput = options.mirrorInput ?? true;
    this.wasmBaseUrl = options.wasmBaseUrl ?? DEFAULT_WASM_BASE_URL;
    this.modelAssetPath = options.modelAssetPath ?? DEFAULT_MODEL_ASSET_PATH;
  }

  async start() {
    if (this.running) return;
    this.running = true;
    this.lastInferenceMs = 0;
    this.lastVideoTime = -1;
    this.lastSwipeMs = 0;
    this.lastPinchMs = 0;
    this.hasHand = false;
    this.palmSamples = [];

    try {
      this.setStatus('requesting_camera', '正在请求前置摄像头');
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 424 },
          height: { ideal: 240 },
          frameRate: { ideal: 15, max: 20 },
        },
        audio: false,
      });

      this.video.srcObject = this.stream;
      this.video.muted = true;
      this.video.playsInline = true;
      await this.video.play();

      this.setStatus('loading_model', '正在加载手势识别模型');
      const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision');
      const vision = await FilesetResolver.forVisionTasks(this.wasmBaseUrl);
      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: this.modelAssetPath,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.5,
      });

      this.setStatus('running', '识别中');
      this.onAction({ type: 'ready' });
      this.tick();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error || '手势识别启动失败');
      this.setStatus('error', message);
      this.onAction({ type: 'error', message });
      this.stop();
      throw error;
    }
  }

  stop() {
    this.running = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = 0;
    }

    if (this.handLandmarker) {
      this.handLandmarker.close();
      this.handLandmarker = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    this.video.pause();
    this.video.removeAttribute('src');
    this.video.srcObject = null;
    this.hasHand = false;
    this.palmSamples = [];
    this.lastStatus = null;
    this.setStatus('idle', '未开启');
  }

  private tick = () => {
    if (!this.running) return;
    const now = performance.now();
    if (now - this.lastInferenceMs >= this.inferenceIntervalMs) {
      this.lastInferenceMs = now;
      this.detect(now);
    }
    this.animationFrameId = requestAnimationFrame(this.tick);
  };

  private detect(nowMs: number) {
    if (!this.handLandmarker || this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    if (this.video.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = this.video.currentTime;

    let result: HandLandmarkerResult | null = null;
    try {
      result = this.handLandmarker.detectForVideo(this.video, nowMs);
    } catch (error) {
      const message = error instanceof Error ? error.message : '手势识别推理失败';
      this.setStatus('error', message);
      this.onAction({ type: 'error', message });
      return;
    }

    const landmarks = result.landmarks?.[0];
    if (!landmarks || landmarks.length < 21) {
      if (this.hasHand) {
        this.hasHand = false;
        this.palmSamples = [];
        this.setStatus('lost_hand', '未检测到手');
        this.onAction({ type: 'lost_hand' });
      }
      return;
    }

    const openness = computeOpenness(landmarks);
    const palmX = computePalmVisualX(landmarks, this.mirrorInput);
    const palmY = computePalmY(landmarks);
    this.hasHand = true;
    this.setStatus('hand_present', '已识别手掌');
    this.pushPalmSample(nowMs, palmX, palmY, openness);

    const swipe = this.detectSwipe(nowMs);
    if (swipe) {
      this.logDebug(nowMs, palmX, palmY, openness, swipe.type);
      this.onAction(swipe);
      return;
    }

    if (!this.hasStrongHorizontalMotion()) {
      const openClose = this.detectOpenClose(nowMs, openness);
      if (openClose) {
        this.logDebug(nowMs, palmX, palmY, openness, openClose.type);
        this.onAction(openClose);
        return;
      }
    }

    this.logDebug(nowMs, palmX, palmY, openness, 'none');
  }

  private pushPalmSample(nowMs: number, palmX: number, palmY: number, openness: number) {
    this.palmSamples.push({ timeMs: nowMs, x: palmX, y: palmY, openness });
    this.palmSamples = this.palmSamples.filter((sample) => nowMs - sample.timeMs <= 320);
  }

  private getSwipeWindowMetrics() {
    const first = this.palmSamples[0];
    const last = this.palmSamples[this.palmSamples.length - 1];
    if (!first || !last) {
      return { dx: 0, dy: 0, vx: 0, dt: 0 };
    }

    const dt = Math.max((last.timeMs - first.timeMs) / 1000, 0.001);
    const dx = last.x - first.x;
    const dy = last.y - first.y;
    return { dx, dy, vx: dx / dt, dt };
  }

  private detectSwipe(nowMs: number): AirGestureAction | null {
    if (nowMs - this.lastSwipeMs < this.swipeCooldownMs) return null;
    if (this.palmSamples.length < 4) return null;

    const { dx, dy, vx } = this.getSwipeWindowMetrics();
    const minDx = 0.1;
    const minVx = 0.32;
    const maxDy = 0.18;
    if (Math.abs(dx) < minDx) return null;
    if (Math.abs(vx) < minVx) return null;
    if (Math.abs(dy) > maxDy) return null;

    this.lastSwipeMs = nowMs;
    this.palmSamples = [];
    const strength = Math.min(2, Math.max(0.75, Math.abs(dx) / minDx));
    // visualX 按用户看到的方向归一化：变小是向左，变大是向右。
    return dx < 0
      ? { type: 'swipe_left', strength }
      : { type: 'swipe_right', strength };
  }

  private hasStrongHorizontalMotion() {
    if (this.palmSamples.length < 3) return false;
    const { dx } = this.getSwipeWindowMetrics();
    return Math.abs(dx) > 0.06;
  }

  private detectOpenClose(nowMs: number, openness: number): AirGestureAction | null {
    if (nowMs - this.lastPinchMs < this.pinchCooldownMs) return null;

    const openThreshold = 2.15;
    const closeThreshold = 1.65;
    if (openness >= openThreshold) {
      this.lastPinchMs = nowMs;
      return { type: 'open', amount: Math.min(1.8, openness / openThreshold) };
    } else if (openness <= closeThreshold && openness > 0) {
      this.lastPinchMs = nowMs;
      return { type: 'close', amount: Math.min(1.8, closeThreshold / Math.max(openness, 0.1)) };
    }
    return null;
  }

  private setStatus(status: AirGestureStatus, message?: string) {
    if (this.lastStatus === status) return;
    this.lastStatus = status;
    this.onStatus?.(status, message);
  }

  private logDebug(nowMs: number, palmX: number, palmY: number, openness: number, action: string) {
    if (!this.debug || nowMs - this.lastDebugMs < 250) return;
    this.lastDebugMs = nowMs;
    const { dx, dy, vx } = this.getSwipeWindowMetrics();
    const cooldownLeft = Math.max(0, this.swipeCooldownMs - (nowMs - this.lastSwipeMs));
    console.debug('[AirGesture]', {
      handPresent: this.hasHand,
      palmX: Number(palmX.toFixed(3)),
      palmY: Number(palmY.toFixed(3)),
      dxWindow: Number(dx.toFixed(3)),
      dyWindow: Number(dy.toFixed(3)),
      vx: Number(vx.toFixed(3)),
      openness: Number(openness.toFixed(3)),
      action,
      cooldownLeft: Math.round(cooldownLeft),
    });
  }
}
