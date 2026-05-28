import * as THREE from 'three';
import { FIELD_SHADER_CHUNK } from './vector-fields.js';

const FLOAT_PARAMS = {
  type: THREE.FloatType,
  format: THREE.RGBAFormat,
  minFilter: THREE.NearestFilter,
  magFilter: THREE.NearestFilter,
  depthBuffer: false,
  stencilBuffer: false,
};

const TYPE_BYTE_SIZES = {
  char: 1,
  uchar: 1,
  short: 2,
  ushort: 2,
  int: 4,
  uint: 4,
  float: 4,
  double: 8,
};

function detectHeaderEnd(bytes) {
  const marker = 'end_header\n';
  const markerAlt = 'end_header\r\n';
  const decoder = new TextDecoder();
  const probe = decoder.decode(bytes.subarray(0, Math.min(bytes.length, 12000)));
  let idx = probe.indexOf(marker);
  if (idx !== -1) return idx + marker.length;
  idx = probe.indexOf(markerAlt);
  if (idx !== -1) return idx + markerAlt.length;
  throw new Error('PLY header not found');
}

function parsePlyHeader(text) {
  const lines = text.split(/\r?\n/);
  const header = {
    format: 'ascii',
    vertexCount: 0,
    properties: [],
  };
  let currentElement = null;
  for (const line of lines) {
    if (!line) continue;
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'format') header.format = parts[1];
    if (parts[0] === 'element') {
      currentElement = parts[1];
      if (currentElement === 'vertex') {
        header.vertexCount = Number(parts[2]) || 0;
      }
    }
    if (parts[0] === 'property' && currentElement === 'vertex') {
      if (parts[1] === 'list') continue;
      header.properties.push({ type: parts[1], name: parts[2] });
    }
  }
  return header;
}

function parseAsciiVertices(bodyText, header) {
  const lines = bodyText.trim().split(/\r?\n/);
  const positions = new Float32Array(header.vertexCount * 3);
  const colors = new Float32Array(header.vertexCount * 3);
  const xIndex = header.properties.findIndex((p) => p.name === 'x');
  const yIndex = header.properties.findIndex((p) => p.name === 'y');
  const zIndex = header.properties.findIndex((p) => p.name === 'z');
  const rIndex = header.properties.findIndex((p) => p.name === 'red' || p.name === 'r');
  const gIndex = header.properties.findIndex((p) => p.name === 'green' || p.name === 'g');
  const bIndex = header.properties.findIndex((p) => p.name === 'blue' || p.name === 'b');

  for (let i = 0; i < header.vertexCount && i < lines.length; i += 1) {
    const values = lines[i].trim().split(/\s+/);
    positions[i * 3 + 0] = Number(values[xIndex] || 0);
    positions[i * 3 + 1] = Number(values[yIndex] || 0);
    positions[i * 3 + 2] = Number(values[zIndex] || 0);
    colors[i * 3 + 0] = rIndex >= 0 ? Number(values[rIndex]) / 255 : 0.88;
    colors[i * 3 + 1] = gIndex >= 0 ? Number(values[gIndex]) / 255 : 0.76;
    colors[i * 3 + 2] = bIndex >= 0 ? Number(values[bIndex]) / 255 : 1.0;
  }

  return { positions, colors };
}

function readValue(view, offset, type) {
  if (type === 'char') return view.getInt8(offset);
  if (type === 'uchar') return view.getUint8(offset);
  if (type === 'short') return view.getInt16(offset, true);
  if (type === 'ushort') return view.getUint16(offset, true);
  if (type === 'int') return view.getInt32(offset, true);
  if (type === 'uint') return view.getUint32(offset, true);
  if (type === 'double') return view.getFloat64(offset, true);
  return view.getFloat32(offset, true);
}

function parseBinaryVertices(buffer, dataOffset, header) {
  const positions = new Float32Array(header.vertexCount * 3);
  const colors = new Float32Array(header.vertexCount * 3);
  const offsets = [];
  let stride = 0;
  header.properties.forEach((property) => {
    offsets.push(stride);
    stride += TYPE_BYTE_SIZES[property.type] || 4;
  });

  const xIndex = header.properties.findIndex((p) => p.name === 'x');
  const yIndex = header.properties.findIndex((p) => p.name === 'y');
  const zIndex = header.properties.findIndex((p) => p.name === 'z');
  const rIndex = header.properties.findIndex((p) => p.name === 'red' || p.name === 'r');
  const gIndex = header.properties.findIndex((p) => p.name === 'green' || p.name === 'g');
  const bIndex = header.properties.findIndex((p) => p.name === 'blue' || p.name === 'b');
  const view = new DataView(buffer, dataOffset);

  for (let i = 0; i < header.vertexCount; i += 1) {
    const rowOffset = i * stride;
    positions[i * 3 + 0] = readValue(view, rowOffset + offsets[xIndex], header.properties[xIndex]?.type || 'float');
    positions[i * 3 + 1] = readValue(view, rowOffset + offsets[yIndex], header.properties[yIndex]?.type || 'float');
    positions[i * 3 + 2] = readValue(view, rowOffset + offsets[zIndex], header.properties[zIndex]?.type || 'float');
    colors[i * 3 + 0] = rIndex >= 0 ? readValue(view, rowOffset + offsets[rIndex], header.properties[rIndex].type) / 255 : 0.88;
    colors[i * 3 + 1] = gIndex >= 0 ? readValue(view, rowOffset + offsets[gIndex], header.properties[gIndex].type) / 255 : 0.76;
    colors[i * 3 + 2] = bIndex >= 0 ? readValue(view, rowOffset + offsets[bIndex], header.properties[bIndex].type) / 255 : 1.0;
  }

  return { positions, colors };
}

function normalizeVertices(positions) {
  const min = new THREE.Vector3(Infinity, Infinity, Infinity);
  const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
  const center = new THREE.Vector3();
  const probe = new THREE.Vector3();
  for (let i = 0; i < positions.length; i += 3) {
    probe.set(positions[i], positions[i + 1], positions[i + 2]);
    min.min(probe);
    max.max(probe);
  }
  center.addVectors(min, max).multiplyScalar(0.5);
  const size = new THREE.Vector3().subVectors(max, min);
  const scale = 2.0 / Math.max(size.x, size.y, size.z, 1e-5);
  const normalized = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    normalized[i] = (positions[i] - center.x) * scale;
    normalized[i + 1] = (positions[i + 1] - center.y) * scale;
    normalized[i + 2] = (positions[i + 2] - center.z) * scale;
  }
  return {
    positions: normalized,
    center,
    size,
    radius: Math.max(size.length() * 0.25 * scale, 0.65),
  };
}

function createFallbackCloud(count = 48000) {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const t = i / count;
    const spiral = 18.0 * Math.PI * t;
    const arm = 0.25 + 0.75 * Math.sqrt(t);
    const wobble = Math.sin(spiral * 3.0) * 0.12;
    positions[i * 3 + 0] = Math.cos(spiral) * arm + wobble;
    positions[i * 3 + 1] = (t - 0.5) * 1.8 + Math.sin(spiral * 0.5) * 0.25;
    positions[i * 3 + 2] = Math.sin(spiral) * arm;
    colors[i * 3 + 0] = 0.65 + 0.35 * Math.sin(t * Math.PI);
    colors[i * 3 + 1] = 0.45 + 0.35 * Math.cos(t * Math.PI * 2.0);
    colors[i * 3 + 2] = 0.82 + 0.18 * Math.sin(spiral * 0.25);
  }
  return {
    ...normalizeVertices(positions),
    colors,
  };
}

export class DataPipeline {
  constructor({ renderer }) {
    this.renderer = renderer;
    this.textureSize = 0;
    this.count = 0;
    this.bounds = {
      center: new THREE.Vector3(),
      size: new THREE.Vector3(2, 2, 2),
      radius: 1,
    };
    this.simulationScene = new THREE.Scene();
    this.simulationCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.simulationQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.simulationScene.add(this.simulationQuad);
    this.points = null;
    this.state = {
      positionIn: null,
      velocityIn: null,
      positionOut: null,
      velocityOut: null,
    };
  }

  async load(url, options = {}) {
    const lowerUrl = String(url || '').toLowerCase();
    const isSplatLike = lowerUrl.endsWith('.splat') || lowerUrl.endsWith('.ksplat');
    const allowSplatFallback = options.allowSplatFallback === true;
    let normalizedData = null;
    let colors = null;

    if (url && !isSplatLike) {
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`PLY request failed: ${response.status}`);
        const buffer = await response.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        const headerEnd = detectHeaderEnd(bytes);
        const headerText = new TextDecoder().decode(bytes.subarray(0, headerEnd));
        const header = parsePlyHeader(headerText);
        const parsed = header.format.startsWith('binary')
          ? parseBinaryVertices(buffer, headerEnd, header)
          : parseAsciiVertices(new TextDecoder().decode(bytes.subarray(headerEnd)), header);
        colors = parsed.colors;
        normalizedData = normalizeVertices(parsed.positions);
      } catch (error) {
        console.warn('[BrainDance][DataPipeline] PLY load failed, using fallback cloud.', error);
      }
    }

    if (isSplatLike) {
      console.info('[BrainDance][DataPipeline] Skip PLY parsing for splat source:', url);
    }

    if (!normalizedData && (!isSplatLike || allowSplatFallback)) {
      const fallback = createFallbackCloud();
      normalizedData = fallback;
      colors = fallback.colors;
    }

    if (!normalizedData) {
      this.bounds.center.set(0, 0, 0);
      this.bounds.size.set(0, 0, 0);
      this.bounds.radius = 0;
      this.count = 0;
      this.textureSize = 0;
      this.points = null;
      return;
    }

    this.bounds.center.copy(normalizedData.center);
    this.bounds.size.copy(normalizedData.size);
    this.bounds.radius = normalizedData.radius;
    this.count = normalizedData.positions.length / 3;
    this.textureSize = Math.ceil(Math.sqrt(this.count));

    const total = this.textureSize * this.textureSize;
    const positionData = new Float32Array(total * 4);
    const velocityData = new Float32Array(total * 4);
    const colorData = new Float32Array(total * 3);
    const refs = new Float32Array(total * 2);
    const sizes = new Float32Array(total);

    for (let i = 0; i < total; i += 1) {
      const idx3 = i * 3;
      const idx4 = i * 4;
      const u = (i % this.textureSize) / Math.max(this.textureSize - 1, 1);
      const v = Math.floor(i / this.textureSize) / Math.max(this.textureSize - 1, 1);
      refs[i * 2 + 0] = u;
      refs[i * 2 + 1] = v;
      sizes[i] = i < this.count ? 1.0 + (i % 17) * 0.05 : 0.0;
      if (i < this.count) {
        positionData[idx4 + 0] = normalizedData.positions[idx3 + 0];
        positionData[idx4 + 1] = normalizedData.positions[idx3 + 1];
        positionData[idx4 + 2] = normalizedData.positions[idx3 + 2];
        positionData[idx4 + 3] = 1.0;
        velocityData[idx4 + 0] = 0.0;
        velocityData[idx4 + 1] = 0.0;
        velocityData[idx4 + 2] = 0.0;
        velocityData[idx4 + 3] = Math.random();
        colorData[idx3 + 0] = colors[idx3 + 0];
        colorData[idx3 + 1] = colors[idx3 + 1];
        colorData[idx3 + 2] = colors[idx3 + 2];
      }
    }

    this.initialPositionTexture = new THREE.DataTexture(positionData, this.textureSize, this.textureSize, THREE.RGBAFormat, THREE.FloatType);
    this.initialPositionTexture.needsUpdate = true;
    this.initialVelocityTexture = new THREE.DataTexture(velocityData, this.textureSize, this.textureSize, THREE.RGBAFormat, THREE.FloatType);
    this.initialVelocityTexture.needsUpdate = true;

    this.positionsTextureA = new THREE.WebGLRenderTarget(this.textureSize, this.textureSize, FLOAT_PARAMS);
    this.positionsTextureB = new THREE.WebGLRenderTarget(this.textureSize, this.textureSize, FLOAT_PARAMS);
    this.velocityTextureA = new THREE.WebGLRenderTarget(this.textureSize, this.textureSize, FLOAT_PARAMS);
    this.velocityTextureB = new THREE.WebGLRenderTarget(this.textureSize, this.textureSize, FLOAT_PARAMS);

    this.seedMaterial = new THREE.ShaderMaterial({
      uniforms: { seedTexture: { value: null } },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D seedTexture;
        varying vec2 vUv;
        void main() {
          gl_FragColor = texture2D(seedTexture, vUv);
        }
      `,
    });

    this.velocityMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uPositions: { value: this.initialPositionTexture },
        uVelocity: { value: this.initialVelocityTexture },
        uDelta: { value: 0.016 },
        uTime: { value: 0 },
        uProgress: { value: 0 },
        uPinch: { value: 0 },
        uFieldMode: { value: 0 },
        uDamping: { value: 0.93 },
      },
      vertexShader: this.seedMaterial.vertexShader,
      fragmentShader: `
        precision highp float;
        varying vec2 vUv;
        uniform sampler2D uPositions;
        uniform sampler2D uVelocity;
        uniform float uDelta;
        uniform float uTime;
        uniform float uProgress;
        uniform float uPinch;
        uniform float uFieldMode;
        uniform float uDamping;
        ${FIELD_SHADER_CHUNK}
        void main() {
          vec3 pos = texture2D(uPositions, vUv).xyz;
          vec3 vel = texture2D(uVelocity, vUv).xyz;
          vec3 accel = evaluateField(pos, vel, uTime, uProgress, uPinch, uFieldMode);
          vel = (vel + accel * uDelta) * pow(uDamping, uDelta * 60.0);
          gl_FragColor = vec4(vel, 1.0);
        }
      `,
    });

    this.positionMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uPositions: { value: this.initialPositionTexture },
        uVelocity: { value: this.initialVelocityTexture },
        uDelta: { value: 0.016 },
        uTime: { value: 0 },
      },
      vertexShader: this.seedMaterial.vertexShader,
      fragmentShader: `
        precision highp float;
        varying vec2 vUv;
        uniform sampler2D uPositions;
        uniform sampler2D uVelocity;
        uniform float uDelta;
        uniform float uTime;
        void main() {
          vec4 current = texture2D(uPositions, vUv);
          vec3 vel = texture2D(uVelocity, vUv).xyz;
          vec3 pos = current.xyz + vel * uDelta;
          float leash = 2.6 + sin(uTime * 0.11) * 0.2;
          float lenPos = length(pos);
          if (lenPos > leash) {
            pos = normalize(pos) * leash;
          }
          gl_FragColor = vec4(pos, current.w);
        }
      `,
    });

    this._seedTarget(this.positionsTextureA, this.initialPositionTexture);
    this._seedTarget(this.positionsTextureB, this.initialPositionTexture);
    this._seedTarget(this.velocityTextureA, this.initialVelocityTexture);
    this._seedTarget(this.velocityTextureB, this.initialVelocityTexture);

    this.state.positionIn = this.positionsTextureA;
    this.state.positionOut = this.positionsTextureB;
    this.state.velocityIn = this.velocityTextureA;
    this.state.velocityOut = this.velocityTextureB;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('aRef', new THREE.BufferAttribute(refs, 2));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(colorData, 3));
    geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
    geometry.setDrawRange(0, this.count);

    this.renderMaterial = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uPositions: { value: this.state.positionIn.texture },
        uProgress: { value: 0 },
        uFieldMode: { value: 0 },
        uTime: { value: 0 },
        uPointScale: { value: 1.0 },
        uViewportHeight: { value: 1080 },
        uVisibility: { value: 0.18 },
      },
      vertexShader: `
        precision highp float;
        attribute vec2 aRef;
        attribute vec3 aColor;
        attribute float aSize;
        uniform sampler2D uPositions;
        uniform float uProgress;
        uniform float uFieldMode;
        uniform float uTime;
        uniform float uPointScale;
        uniform float uViewportHeight;
        uniform float uVisibility;
        varying vec3 vColor;
        varying float vAlpha;
        varying float vModeBand;
        void main() {
          vec3 pos = texture2D(uPositions, aRef).xyz;
          vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
          float depthFade = clamp(1.7 / max(-mvPosition.z, 0.001), 0.0, 1.0);
          float modeBand = mod(uFieldMode, 6.0);
          float modePulse = 0.86 + 0.22 * sin(uTime * (0.8 + modeBand * 0.17) + aRef.x * 11.0 + aRef.y * 7.0);
          float size = (uPointScale + aSize * (0.18 + uProgress * 0.48 + modeBand * 0.035)) * depthFade * 10.0 * modePulse;
          gl_PointSize = clamp(size, 1.2, 16.0);
          gl_Position = projectionMatrix * mvPosition;
          vColor = aColor;
          vAlpha = depthFade * uVisibility * modePulse;
          vModeBand = modeBand;
        }
      `,
      fragmentShader: `
        precision highp float;
        varying vec3 vColor;
        varying float vAlpha;
        varying float vModeBand;
        void main() {
          vec2 p = gl_PointCoord - 0.5;
          float dist = dot(p, p);
          float mask = smoothstep(0.25, 0.0, dist);
          vec3 modeTint = vec3(
            0.45 + 0.08 * vModeBand,
            0.56 + 0.05 * sin(vModeBand * 1.2),
            0.62 + 0.06 * cos(vModeBand * 0.9)
          );
          vec3 color = mix(vColor, modeTint, 0.38);
          gl_FragColor = vec4(color, mask * vAlpha);
        }
      `,
    });

    this.points = new THREE.Points(geometry, this.renderMaterial);
    this.points.frustumCulled = false;
  }

  _seedTarget(target, sourceTexture) {
    this.seedMaterial.uniforms.seedTexture.value = sourceTexture;
    this.simulationQuad.material = this.seedMaterial;
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.simulationScene, this.simulationCamera);
    this.renderer.setRenderTarget(null);
  }

  step({ dt, time, progress, pinch, fieldMode, viewportHeight }) {
    if (!this.points) return;
    this.velocityMaterial.uniforms.uPositions.value = this.state.positionIn.texture;
    this.velocityMaterial.uniforms.uVelocity.value = this.state.velocityIn.texture;
    this.velocityMaterial.uniforms.uDelta.value = dt;
    this.velocityMaterial.uniforms.uTime.value = time;
    this.velocityMaterial.uniforms.uProgress.value = progress;
    this.velocityMaterial.uniforms.uPinch.value = pinch;
    this.velocityMaterial.uniforms.uFieldMode.value = fieldMode;
    this.simulationQuad.material = this.velocityMaterial;
    this.renderer.setRenderTarget(this.state.velocityOut);
    this.renderer.render(this.simulationScene, this.simulationCamera);

    this.positionMaterial.uniforms.uPositions.value = this.state.positionIn.texture;
    this.positionMaterial.uniforms.uVelocity.value = this.state.velocityOut.texture;
    this.positionMaterial.uniforms.uDelta.value = dt;
    this.positionMaterial.uniforms.uTime.value = time;
    this.simulationQuad.material = this.positionMaterial;
    this.renderer.setRenderTarget(this.state.positionOut);
    this.renderer.render(this.simulationScene, this.simulationCamera);
    this.renderer.setRenderTarget(null);

    [this.state.positionIn, this.state.positionOut] = [this.state.positionOut, this.state.positionIn];
    [this.state.velocityIn, this.state.velocityOut] = [this.state.velocityOut, this.state.velocityIn];
    this.renderMaterial.uniforms.uPositions.value = this.state.positionIn.texture;
    this.renderMaterial.uniforms.uProgress.value = progress;
    this.renderMaterial.uniforms.uFieldMode.value = fieldMode;
    this.renderMaterial.uniforms.uTime.value = time;
    this.renderMaterial.uniforms.uViewportHeight.value = viewportHeight;
  }

  reset() {
    if (!this.initialPositionTexture) return;
    this._seedTarget(this.positionsTextureA, this.initialPositionTexture);
    this._seedTarget(this.positionsTextureB, this.initialPositionTexture);
    this._seedTarget(this.velocityTextureA, this.initialVelocityTexture);
    this._seedTarget(this.velocityTextureB, this.initialVelocityTexture);
    this.state.positionIn = this.positionsTextureA;
    this.state.positionOut = this.positionsTextureB;
    this.state.velocityIn = this.velocityTextureA;
    this.state.velocityOut = this.velocityTextureB;
    this.renderMaterial.uniforms.uPositions.value = this.state.positionIn.texture;
  }

  dispose() {
    this.points?.geometry?.dispose();
    this.points?.material?.dispose();
    this.positionsTextureA?.dispose();
    this.positionsTextureB?.dispose();
    this.velocityTextureA?.dispose();
    this.velocityTextureB?.dispose();
    this.initialPositionTexture?.dispose();
    this.initialVelocityTexture?.dispose();
    this.seedMaterial?.dispose();
    this.velocityMaterial?.dispose();
    this.positionMaterial?.dispose();
    this.simulationQuad?.geometry?.dispose();
  }
}
