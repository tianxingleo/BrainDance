// 4. Data Pipeline & GPGPU Core (主控引擎)
// 弃用 CPU 遍历, 构建 Ping-pong FBO 的核心渲染管线

import * as THREE from 'three';
import { GPUComputationRenderer } from 'three/addons/misc/GPUComputationRenderer.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

import { simFragmentShader, renderVertexShader, renderFragmentShader } from './vector_fields.js';
import { CinematicController } from './cinematic.js';
import { IOPostController } from './io_post.js';

class SparkArtEngine {
    constructor() {
        // --- 核心 WebGL2 环境 ---
        this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000);
        document.body.appendChild(this.renderer.domElement);

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 0, 5);

        this.clock = new THREE.Clock();

        // Modules instantiation
        this.cinematic = new CinematicController(this.camera, this.scene, this.renderer);
        this.ioPost = new IOPostController(this.scene, this.camera, this.renderer);

        // 绑定相机序列开关
        document.getElementById('ui-cam-trigger').addEventListener('click', () => {
            this.cinematic.triggerNextSequence();
        });

        // Resize
        window.addEventListener('resize', this.onWindowResize.bind(this));

        // Start Data Pipeline
        this.initDataPipeline();

        // 初始运镜
        this.cinematic.triggerNextSequence();
    }

    async initDataPipeline() {
        console.log("Loading PLY 3DGS Cloud Topology...");
        let WIDTH = 512;
        let posArray = null;
        let colArray = null;

        try {
            const loader = new PLYLoader();
            const geometry = await loader.loadAsync('./input_3dgs.ply');

            const posAttr = geometry.getAttribute('position');
            // 尝试获取标准 color 或 3DGS 特有的 f_dc_* 属性
            let colAttr = geometry.getAttribute('color');
            const fdc0 = geometry.getAttribute('f_dc_0');
            const fdc1 = geometry.getAttribute('f_dc_1');
            const fdc2 = geometry.getAttribute('f_dc_2');

            const vertexCount = posAttr.count;
            WIDTH = Math.ceil(Math.sqrt(vertexCount));
            if (WIDTH < 512) WIDTH = 512;

            posArray = new Float32Array(WIDTH * WIDTH * 4).fill(0.0);
            colArray = new Float32Array(WIDTH * WIDTH * 3).fill(0.8);

            let minX = Infinity, minY = Infinity, minZ = Infinity;
            let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

            const SH_C0 = 0.28209479177387814;

            for (let i = 0; i < vertexCount; i++) {
                let fx = posAttr.getX(i);
                let fy = posAttr.getY(i);
                let fz = posAttr.getZ(i);

                minX = Math.min(minX, fx); minY = Math.min(minY, fy); minZ = Math.min(minZ, fz);
                maxX = Math.max(maxX, fx); maxY = Math.max(maxY, fy); maxZ = Math.max(maxZ, fz);

                posArray[i * 4 + 0] = fx;
                posArray[i * 4 + 1] = fy;
                posArray[i * 4 + 2] = fz;
                posArray[i * 4 + 3] = 1.0;

                if (colAttr) {
                    colArray[i * 3 + 0] = colAttr.getX(i);
                    colArray[i * 3 + 1] = colAttr.getY(i);
                    colArray[i * 3 + 2] = colAttr.getZ(i);
                } else if (fdc0) {
                    // Convert SH0 to RGB
                    colArray[i * 3 + 0] = 0.5 + SH_C0 * fdc0.getX(i);
                    colArray[i * 3 + 1] = 0.5 + SH_C0 * fdc1.getX(i);
                    colArray[i * 3 + 2] = 0.5 + SH_C0 * fdc2.getX(i);
                }
            }

            // 归一化 BBox
            let cx = (minX + maxX) / 2;
            let cy = (minY + maxY) / 2;
            let cz = (minZ + maxZ) / 2;
            let maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
            let scale = maxDim === 0 ? 1 : (3.0 / maxDim);

            for (let i = 0; i < vertexCount; i++) {
                posArray[i * 4 + 0] = (posArray[i * 4 + 0] - cx) * scale;
                posArray[i * 4 + 1] = (posArray[i * 4 + 1] - cy) * scale;
                posArray[i * 4 + 2] = (posArray[i * 4 + 2] - cz) * scale;
            }

            // 校准运镜系统边界
            this.cinematic.targetBox.set(
                new THREE.Vector3(-1.5, -1.5, -1.5),
                new THREE.Vector3(1.5, 1.5, 1.5)
            );

            console.log(`Successfully mapped ${vertexCount} PLY nodes into ${WIDTH}x${WIDTH} GPU Tensor.`);

        } catch (e) {
            console.warn("Failed loading PLY, using fallback generative spheres...", e);
            WIDTH = 512;
            posArray = new Float32Array(WIDTH * WIDTH * 4).fill(0.0);
            colArray = new Float32Array(WIDTH * WIDTH * 3).fill(0.8);
            for (let i = 0; i < WIDTH * WIDTH * 4; i += 4) {
                const r = 2.0 * Math.cbrt(Math.random());
                const theta = Math.random() * 2 * Math.PI;
                const phi = Math.acos(2 * Math.random() - 1);
                posArray[i] = r * Math.sin(phi) * Math.cos(theta);
                posArray[i + 1] = r * Math.sin(phi) * Math.sin(theta);
                posArray[i + 2] = r * Math.cos(phi);
                posArray[i + 3] = 1.0;
            }
        }

        this.initGPGPU(WIDTH, posArray, colArray);
        this.animate();
    }

    initGPGPU(WIDTH, posArray, colArray) {
        this.gpgpu = new GPUComputationRenderer(WIDTH, WIDTH, this.renderer);
        if (this.renderer.capabilities.isWebGL2 === false) {
            this.gpgpu.setDataType(THREE.HalfFloatType);
        }

        const dtPosition = this.gpgpu.createTexture();
        const dtVelocity = this.gpgpu.createTexture();
        const dtOrigin = this.gpgpu.createTexture();

        dtPosition.image.data.set(posArray);
        dtOrigin.image.data.set(posArray);

        // Setup variables (Ping-pong FBOs)
        this.velVariable = this.gpgpu.addVariable('textureVelocity', simFragmentShader, dtVelocity);
        this.posVariable = this.gpgpu.addVariable('texturePosition', simFragmentShader, dtPosition);

        // Core Shader Inject 
        this.gpgpu.setVariableDependencies(this.velVariable, [this.posVariable, this.velVariable]);
        this.gpgpu.setVariableDependencies(this.posVariable, [this.posVariable, this.velVariable]);

        // Uniforms for physics logic mapping
        const uniforms = {
            uTime: { value: 0 },
            uProgress: { value: 0 },
            uVectorFieldType: { value: 0 },
            textureOriginPosition: { value: dtOrigin }
        };

        this.velVariable.material.uniforms = uniforms;
        this.posVariable.material.uniforms = uniforms;

        const error = this.gpgpu.init();
        if (error !== null) console.error(error);

        // --- Build Particle System Geometry ---
        const geometry = new THREE.BufferGeometry();
        const indices = new Float32Array(WIDTH * WIDTH * 3);
        const colors = new Float32Array(WIDTH * WIDTH * 3);

        for (let i = 0, j = 0; i < WIDTH * WIDTH; i++, j += 3) {
            let x = (i % WIDTH) / WIDTH;
            let y = Math.floor(i / WIDTH) / WIDTH;
            indices[j] = x; indices[j + 1] = y; indices[j + 2] = 0;

            colors[j] = colArray[j] || 1.0;
            colors[j + 1] = colArray[j + 1] || 1.0;
            colors[j + 2] = colArray[j + 2] || 1.0;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(indices, 3));
        geometry.setAttribute('aColor', new THREE.BufferAttribute(colors, 3));

        // Render Shader Material Setup
        this.particleMaterial = new THREE.ShaderMaterial({
            uniforms: {
                texturePosition: { value: null },
                uProgress: { value: 0.0 }
            },
            vertexShader: renderVertexShader,
            fragmentShader: renderFragmentShader,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        this.particleMesh = new THREE.Points(geometry, this.particleMaterial);

        // Bind dynamic box fitting to our Frustum seq target Box
        // 实际上可以动态计算所有粒子的最大包围盒，此处简化为固有约束
        this.scene.add(this.particleMesh);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.ioPost.resize(window.innerWidth, window.innerHeight);
    }

    animate(ts) {
        requestAnimationFrame(this.animate.bind(this));

        let delta = this.clock.getDelta();
        let time = this.clock.getElapsedTime();

        // 抽取前端数据双向绑定到底层 Core 显存 Uniforms
        const vfType = this.ioPost.vjVectorFieldId;
        const progress = this.ioPost.targetUProgress; // Not lerped for logical step

        if (this.gpgpu) {
            this.velVariable.material.uniforms.uTime.value = time;
            this.velVariable.material.uniforms.uProgress.value = progress;
            this.velVariable.material.uniforms.uVectorFieldType.value = vfType;

            this.posVariable.material.uniforms.uTime.value = time;
            this.posVariable.material.uniforms.uProgress.value = progress;
            this.posVariable.material.uniforms.uVectorFieldType.value = vfType;

            this.gpgpu.compute();

            // Feed updated Ping-pong textures into Render Shader
            this.particleMaterial.uniforms.texturePosition.value = this.gpgpu.getCurrentRenderTarget(this.posVariable).texture;
            this.particleMaterial.uniforms.uProgress.value = this.ioPost.uProgress; // Lerped for visuals
        }

        // 运镜焦点设置在粒子分布中心 (此处假设原点)
        this.cinematic.setFocusPoint(0, 0, 0);

        // 调用自定义 IO Post 进行后期重映射合成渲染输出
        this.ioPost.update(delta);
    }
}

// Boot Engine
console.log("⚡ Booting Spark Generative Art Engine...");
window.sparkEngine = new SparkArtEngine();
