export type GestureAction =
  | { type: 'look'; dx: number; dy: number }
  | { type: 'pinch'; scaleDelta: number }
  | { type: 'pan'; dx: number; dy: number }
  | { type: 'longpress'; x: number; y: number }
  | { type: 'doubletap'; x: number; y: number }
  | { type: 'swipe_forward' }
  | { type: 'swipe_backward' };

export interface GestureConfig {
  onAction: (action: GestureAction) => void;
  longPressDelay?: number;
  doubleTapInterval?: number;
  swipeThreshold?: number;
}

/**
 * Unified touch + mouse gesture handler.
 *
 * Touch:
 *   1-finger drag → look
 *   2-finger pinch → zoom (dolly)
 *   2-finger drag  → pan
 *   long press     → focus / anchor
 *   fast swipe up/down → navigate pose graph
 *   double tap     → reset view
 *
 * Mouse:
 *   left drag      → look
 *   right/mid drag  → pan
 *   wheel          → zoom
 *   double click   → reset view
 */
export class GestureHandler {
  private el: HTMLElement;
  private cfg: Required<GestureConfig>;

  // Touch state
  private pts = new Map<number, { x: number; y: number }>();
  private lastSingle: { x: number; y: number } | null = null;
  private initPinchDist = 0;
  private dragging = false;
  private pinching = false;

  // Long press
  private lpTimer: ReturnType<typeof setTimeout> | null = null;
  private lpStart = { x: 0, y: 0 };

  // Double tap
  private lastTapTime = 0;
  private lastTapPos = { x: 0, y: 0 };

  // Swipe
  private swipeStart = { x: 0, y: 0 };
  private swipeTime = 0;

  // Mouse
  private mouseDown = false;
  private lastMouse = { x: 0, y: 0 };

  constructor(element: HTMLElement, config: GestureConfig) {
    this.el = element;
    this.cfg = {
      onAction: config.onAction,
      longPressDelay: config.longPressDelay ?? 500,
      doubleTapInterval: config.doubleTapInterval ?? 300,
      swipeThreshold: config.swipeThreshold ?? 80,
    };
    this.attach();
  }

  // ── Attach / detach ────────────────────────────────────────

  private attach() {
    this.el.addEventListener('touchstart', this.onTS, { passive: false });
    this.el.addEventListener('touchmove', this.onTM, { passive: false });
    this.el.addEventListener('touchend', this.onTE, { passive: false });
    this.el.addEventListener('touchcancel', this.onTE, { passive: false });
    this.el.addEventListener('mousedown', this.onMD);
    window.addEventListener('mousemove', this.onMM);
    window.addEventListener('mouseup', this.onMU);
    this.el.addEventListener('wheel', this.onWheel, { passive: false });
    this.el.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  detach() {
    this.el.removeEventListener('touchstart', this.onTS);
    this.el.removeEventListener('touchmove', this.onTM);
    this.el.removeEventListener('touchend', this.onTE);
    this.el.removeEventListener('touchcancel', this.onTE);
    this.el.removeEventListener('mousedown', this.onMD);
    window.removeEventListener('mousemove', this.onMM);
    window.removeEventListener('mouseup', this.onMU);
    this.el.removeEventListener('wheel', this.onWheel);
    this.cancelLp();
    this.pts.clear();
  }

  // ── Touch ──────────────────────────────────────────────────

  private onTS = (e: TouchEvent) => {
    e.preventDefault();
    for (let i = 0; i < e.changedTouches.length; i++) {
      const t = e.changedTouches[i];
      this.pts.set(t.identifier, { x: t.clientX, y: t.clientY });
    }

    if (e.touches.length === 1) {
      const t = e.touches[0];
      this.lastSingle = { x: t.clientX, y: t.clientY };
      this.dragging = true;
      this.swipeStart = { x: t.clientX, y: t.clientY };
      this.swipeTime = Date.now();
      this.lpStart = { x: t.clientX, y: t.clientY };
      this.startLp(t.clientX, t.clientY);
    } else if (e.touches.length === 2) {
      this.cancelLp();
      this.dragging = false;
      this.pinching = true;
      const [a, b] = [e.touches[0], e.touches[1]];
      this.initPinchDist = Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY);
    }
  };

  private onTM = (e: TouchEvent) => {
    e.preventDefault();

    if (e.touches.length === 1 && this.dragging) {
      const t = e.touches[0];
      if (this.lastSingle) {
        const dx = t.clientX - this.lastSingle.x;
        const dy = t.clientY - this.lastSingle.y;

        if (Math.hypot(t.clientX - this.lpStart.x, t.clientY - this.lpStart.y) > 10) {
          this.cancelLp();
        }

        this.cfg.onAction({ type: 'look', dx, dy });
      }
      this.lastSingle = { x: t.clientX, y: t.clientY };
    } else if (e.touches.length === 2 && this.pinching) {
      const [a, b] = [e.touches[0], e.touches[1]];
      const dist = Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY);

      if (this.initPinchDist > 0) {
        this.cfg.onAction({ type: 'pinch', scaleDelta: dist - this.initPinchDist });
      }
      this.initPinchDist = dist;
    }
  };

  private onTE = (e: TouchEvent) => {
    e.preventDefault();

    // Swipe detection
    if (this.pts.size === 1 && this.dragging && this.lastSingle) {
      const dy = this.swipeStart.y - this.lastSingle.y;
      const dx = this.swipeStart.x - this.lastSingle.x;
      const elapsed = Date.now() - this.swipeTime;

      if (
        elapsed < 400 &&
        Math.abs(dy) > this.cfg.swipeThreshold &&
        Math.abs(dy) > Math.abs(dx) * 1.5
      ) {
        this.cfg.onAction(dy > 0 ? { type: 'swipe_forward' } : { type: 'swipe_backward' });
      }
    }

    // Double tap
    if (this.pts.size <= 1 && !this.pinching) {
      const t = e.changedTouches[0];
      if (t) {
        const now = Date.now();
        if (
          now - this.lastTapTime < this.cfg.doubleTapInterval &&
          Math.hypot(t.clientX - this.lastTapPos.x, t.clientY - this.lastTapPos.y) < 30
        ) {
          this.cfg.onAction({ type: 'doubletap', x: t.clientX, y: t.clientY });
          this.lastTapTime = 0;
        } else {
          this.lastTapTime = now;
          this.lastTapPos = { x: t.clientX, y: t.clientY };
        }
      }
    }

    for (let i = 0; i < e.changedTouches.length; i++) {
      this.pts.delete(e.changedTouches[i].identifier);
    }
    this.dragging = false;
    this.pinching = false;
    this.lastSingle = null;
    this.cancelLp();
  };

  // ── Mouse ──────────────────────────────────────────────────

  private onMD = (e: MouseEvent) => {
    this.mouseDown = true;
    this.lastMouse = { x: e.clientX, y: e.clientY };
    this.lpStart = { x: e.clientX, y: e.clientY };
    this.startLp(e.clientX, e.clientY);
  };

  private onMM = (e: MouseEvent) => {
    if (!this.mouseDown) return;

    const dx = e.clientX - this.lastMouse.x;
    const dy = e.clientY - this.lastMouse.y;

    if (Math.hypot(e.clientX - this.lpStart.x, e.clientY - this.lpStart.y) > 10) {
      this.cancelLp();
    }

    if (e.buttons === 1) {
      this.cfg.onAction({ type: 'look', dx, dy });
    } else if (e.buttons === 2 || e.buttons === 4) {
      this.cfg.onAction({ type: 'pan', dx, dy });
    }

    this.lastMouse = { x: e.clientX, y: e.clientY };
  };

  private onMU = (e: MouseEvent) => {
    const now = Date.now();
    if (
      now - this.lastTapTime < this.cfg.doubleTapInterval &&
      Math.hypot(e.clientX - this.lastTapPos.x, e.clientY - this.lastTapPos.y) < 30
    ) {
      this.cfg.onAction({ type: 'doubletap', x: e.clientX, y: e.clientY });
      this.lastTapTime = 0;
    } else {
      this.lastTapTime = now;
      this.lastTapPos = { x: e.clientX, y: e.clientY };
    }
    this.mouseDown = false;
    this.cancelLp();
  };

  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    this.cfg.onAction({ type: 'pinch', scaleDelta: -e.deltaY * 0.5 });
  };

  // ── Long press ─────────────────────────────────────────────

  private startLp(x: number, y: number) {
    this.cancelLp();
    this.lpTimer = setTimeout(() => {
      this.cfg.onAction({ type: 'longpress', x, y });
      this.lpTimer = null;
    }, this.cfg.longPressDelay);
  }

  private cancelLp() {
    if (this.lpTimer !== null) {
      clearTimeout(this.lpTimer);
      this.lpTimer = null;
    }
  }

  // ── Cleanup ────────────────────────────────────────────────

  dispose() {
    this.detach();
  }
}
