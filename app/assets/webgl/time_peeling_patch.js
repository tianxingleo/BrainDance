(function () {
  const DEFAULT_MATRIX = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];

  const state = {
    alpha: 0.5,
    mode: 'blend', // blend | base | overlay
    enabled: false,
  };

  function waitFor(checkFn, timeoutMs = 15000, intervalMs = 100) {
    return new Promise((resolve, reject) => {
      const started = Date.now();
      const timer = setInterval(() => {
        try {
          const value = checkFn();
          if (value) {
            clearInterval(timer);
            resolve(value);
            return;
          }
          if (Date.now() - started > timeoutMs) {
            clearInterval(timer);
            reject(new Error('waitFor timeout'));
          }
        } catch (e) {
          clearInterval(timer);
          reject(e);
        }
      }, intervalMs);
    });
  }

  function normalize4(v) {
    const len = Math.hypot(v[0], v[1], v[2], v[3]) || 1;
    return [v[0] / len, v[1] / len, v[2] / len, v[3] / len];
  }

  function decomposeMatrix(matrixArray) {
    const m = Array.isArray(matrixArray) && matrixArray.length === 16 ? matrixArray : DEFAULT_MATRIX;

    const position = [m[12], m[13], m[14]];

    const sx = Math.hypot(m[0], m[1], m[2]) || 1;
    const sy = Math.hypot(m[4], m[5], m[6]) || 1;
    const sz = Math.hypot(m[8], m[9], m[10]) || 1;

    const r00 = m[0] / sx;
    const r01 = m[4] / sy;
    const r02 = m[8] / sz;
    const r10 = m[1] / sx;
    const r11 = m[5] / sy;
    const r12 = m[9] / sz;
    const r20 = m[2] / sx;
    const r21 = m[6] / sy;
    const r22 = m[10] / sz;

    const trace = r00 + r11 + r22;
    let qw, qx, qy, qz;
    if (trace > 0) {
      const s = Math.sqrt(trace + 1.0) * 2;
      qw = 0.25 * s;
      qx = (r21 - r12) / s;
      qy = (r02 - r20) / s;
      qz = (r10 - r01) / s;
    } else if (r00 > r11 && r00 > r22) {
      const s = Math.sqrt(1.0 + r00 - r11 - r22) * 2;
      qw = (r21 - r12) / s;
      qx = 0.25 * s;
      qy = (r01 + r10) / s;
      qz = (r02 + r20) / s;
    } else if (r11 > r22) {
      const s = Math.sqrt(1.0 + r11 - r00 - r22) * 2;
      qw = (r02 - r20) / s;
      qx = (r01 + r10) / s;
      qy = 0.25 * s;
      qz = (r12 + r21) / s;
    } else {
      const s = Math.sqrt(1.0 + r22 - r00 - r11) * 2;
      qw = (r10 - r01) / s;
      qx = (r02 + r20) / s;
      qy = (r12 + r21) / s;
      qz = 0.25 * s;
    }

    return {
      position,
      rotation: normalize4([qx, qy, qz, qw]),
      scale: [sx, sy, sz],
    };
  }

  function getViewer() {
    return window.viewer || null;
  }

  function applyVisuals() {
    if (!state.enabled) return;
    const viewer = getViewer();
    if (!viewer || !viewer.getSplatScene) return;

    const baseScene = viewer.getSplatScene(0);
    const overlayScene = viewer.getSplatScene(1);
    if (!baseScene || !overlayScene) return;

    if (state.mode === 'base') {
      baseScene.visible = true;
      baseScene.opacity = 1.0;
      overlayScene.visible = false;
    } else if (state.mode === 'overlay') {
      baseScene.visible = false;
      overlayScene.visible = true;
      overlayScene.opacity = 1.0;
    } else {
      baseScene.visible = true;
      overlayScene.visible = true;
      baseScene.opacity = Math.max(0.35, 1.0 - state.alpha * 0.65);
      overlayScene.opacity = state.alpha;
    }

    try {
      viewer.update();
      viewer.render();
    } catch (_) {}
  }

  async function loadTimePeelFromFlutter(payload) {
    if (!payload || !payload.base || !payload.overlay) {
      console.error('[TimePeelPatch] invalid payload', payload);
      return;
    }

    if (!window.loadModelFromFlutter) {
      console.error('[TimePeelPatch] loadModelFromFlutter unavailable');
      return;
    }

    state.enabled = true;
    state.alpha = Number(payload.alpha != null ? payload.alpha : 0.5);
    state.mode = 'blend';

    window.loadModelFromFlutter({
      ply: payload.base,
      matrix: payload.pose || null,
    });

    const viewer = await waitFor(() => getViewer(), 20000, 120);
    await waitFor(() => viewer.getSplatScene && viewer.getSplatScene(0), 20000, 120);

    const trs = decomposeMatrix(payload.matrix);

    await viewer.addSplatScene(payload.overlay, {
      showLoadingUI: false,
      progressiveLoad: false,
      position: trs.position,
      rotation: trs.rotation,
      scale: trs.scale,
    });

    applyVisuals();
  }

  window.loadTimePeelFromFlutter = (payload) => {
    loadTimePeelFromFlutter(payload).catch((e) => {
      console.error('[TimePeelPatch] load failed', e);
      state.enabled = false;
    });
  };

  window.setTimePeelAlpha = (alpha) => {
    const v = Number(alpha);
    if (!Number.isFinite(v)) return;
    state.alpha = Math.max(0, Math.min(1, v));
    applyVisuals();
  };

  window.setTimePeelMode = (mode) => {
    if (mode !== 'blend' && mode !== 'base' && mode !== 'overlay') return;
    state.mode = mode;
    applyVisuals();
  };
})();
