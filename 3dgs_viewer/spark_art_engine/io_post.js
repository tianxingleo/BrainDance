// 3. I/O & VJ Performance (状态机与现场交互) + Post-Processing (后期重映射管线)

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { AfterimagePass } from 'three/addons/postprocessing/AfterimagePass.js';

export class IOPostController {
    constructor(scene, camera, renderer) {
        this.renderer = renderer;

        // --- 1. Post Processing 管线 ---
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        this.composer = new EffectComposer(renderer);
        this.composer.addPass(new RenderPass(scene, camera));

        // Afterimage 历史帧采样以构建残影张力
        this.afterimagePass = new AfterimagePass();
        this.afterimagePass.uniforms['damp'].value = 0.9;
        this.composer.addPass(this.afterimagePass);

        // Unreal Bloom 泛光
        this.bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
        this.bloomPass.threshold = 0.2; // 稍微提高阈值，只让极亮区域泛光
        this.bloomPass.strength = 0.8; // 降低强度从 1.6 -> 0.8
        this.bloomPass.radius = 0.4;
        this.composer.addPass(this.bloomPass);

        // --- 2. I/O 状态机 (uProgress, Vector Field Type) ---
        this.uProgress = 0.0; // 初始设为 0，即原始 Splat 拓扑态
        this.targetUProgress = 0.0;
        this.vjVectorFieldId = 0;

        this.initDOM();
        this.initMediaPipe();
    }

    // --- VJ Live Console 绑定 ---
    initDOM() {
        const slider = document.getElementById('ui-progress');
        const selector = document.getElementById('ui-vf-select');

        slider.addEventListener('input', (e) => {
            this.targetUProgress = parseFloat(e.target.value);
        });

        selector.addEventListener('change', (e) => {
            this.vjVectorFieldId = parseInt(e.target.value);
        });
    }

    // --- MediaPipe 手部关键点神经网络解析 Pinch (食指到拇指的距离) ---
    initMediaPipe() {
        const videoElement = document.getElementById('ai-feed');

        // Check if Hands is loaded via CDN globally in index.html
        if (window.Hands) {
            const hands = new window.Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                }
            });

            hands.setOptions({
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            hands.onResults((results) => {
                if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                    const landmarks = results.multiHandLandmarks[0];
                    // Index Finger Tip (8) to Thumb Tip (4)
                    const thumb = landmarks[4];
                    const index = landmarks[8];
                    const dist = Math.hypot(thumb.x - index.x, thumb.y - index.y, thumb.z - index.z);

                    // Normalize Euclidean Distance constraint [0.03 -> 0.25]
                    const minPinch = 0.03;
                    const maxPinch = 0.25;
                    let norm = (dist - minPinch) / (maxPinch - minPinch);
                    norm = Math.max(0, Math.min(1, norm)); // Clamped

                    // 将神经网络输入写入目标进度中，受底层二次平滑系统(Bezier)过滤
                    this.targetUProgress = norm;

                    // 同步反向更新 UI Range
                    document.getElementById('ui-progress').value = norm.toFixed(3);
                }
            });

            // Camera tracking module mapping
            const camera = new window.Camera(videoElement, {
                onFrame: async () => { await hands.send({ image: videoElement }); },
                width: 320, height: 240
            });
            camera.start().catch((e) => console.log('Camera start avoided for local non-https test / auto. Error:', e));
        } else {
            console.error('MediaPipe Hands not initialized. Ensure internet connection to download model.');
        }
    }

    // 每帧渲染调用
    update(dt) {
        // 双向数据绑定的平滑缓冲曲线 (底层 Bezier / Lerp 引擎)
        this.uProgress += (this.targetUProgress - this.uProgress) * Math.min(1.0, dt * 5.0);

        this.composer.render();
    }

    resize(vW, vH) {
        this.composer.setSize(vW, vH);
    }
}
