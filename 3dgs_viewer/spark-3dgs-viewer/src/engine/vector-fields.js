export const FIELD_PRESETS = [
  'Inverse Bloom',
  'Topology Fracture',
  'Polar Stream',
  'Fold Lattice',
  'Curl Ribbon',
  'SDF Implosion',
  'Neural Wake',
  'Torus Shear',
  'Helix Rain',
  'Axis Bloom',
  'Phase Weave',
  'Crystal Drift',
  'Hyper Pulse',
  'Ribbon Collapse',
  'Lotus Gate',
  'Noise Vortex',
  'Magnet Bloom',
  'Wavefront Split',
  'Cerebral Tide',
  'Mobius Fold',
  'Prism Leak',
  'Storm Knot',
  'Vector Choir',
  'Gravity Lace',
  'Mirror Sink',
  'Quantum Bloom',
  'Signal Peel',
  'Synapse Ring',
  'Fractal Jet',
  'Afterimage Drive',
];

export const FIELD_SHADER_CHUNK = /* glsl */`
  #define FIELD_COUNT 30

  vec4 permute(vec4 x){ return mod(((x * 34.0) + 1.0) * x, 289.0); }
  vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }

  float simplexNoise4d(vec4 v) {
    const vec2 C = vec2(0.138196601125011, 0.309016994374947);
    vec4 i = floor(v + dot(v, vec4(0.309016994374947)));
    vec4 x0 = v - i + dot(i, vec4(C.x));
    vec4 i0;
    vec3 isX = step(x0.yzw, x0.xxx);
    vec3 isYZ = step(x0.zww, x0.yyz);
    i0.x = isX.x + isX.y + isX.z;
    i0.yzw = 1.0 - isX;
    i0.y += isYZ.x + isYZ.y;
    i0.zw += 1.0 - isYZ.xy;
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;
    vec4 i3 = clamp(i0, 0.0, 1.0);
    vec4 i2 = clamp(i0 - 1.0, 0.0, 1.0);
    vec4 i1 = clamp(i0 - 2.0, 0.0, 1.0);
    vec4 x1 = x0 - i1 + C.x;
    vec4 x2 = x0 - i2 + C.x * 2.0;
    vec4 x3 = x0 - i3 + C.x * 3.0;
    vec4 x4 = x0 - 1.0 + C.x * 4.0;
    i = mod(i, 289.0);
    float j0 = permute(permute(permute(permute(i.w + vec4(0.0, i1.w, i2.w, i3.w))
      + i.z + vec4(0.0, i1.z, i2.z, i3.z))
      + i.y + vec4(0.0, i1.y, i2.y, i3.y))
      + i.x + vec4(0.0, i1.x, i2.x, i3.x)).x;
    vec4 j1 = permute(permute(permute(permute(i.w + vec4(i1.w, i2.w, i3.w, 1.0))
      + i.z + vec4(i1.z, i2.z, i3.z, 1.0))
      + i.y + vec4(i1.y, i2.y, i3.y, 1.0))
      + i.x + vec4(i1.x, i2.x, i3.x, 1.0));
    vec4 ip = vec4(1.0 / 294.0, 1.0 / 49.0, 1.0 / 7.0, 0.0);

    vec4 p0 = fract(vec4(j0) * ip.xxxx) * 7.0;
    vec4 p1 = fract(j1 * ip.xxxx) * 7.0;
    vec4 p0x = floor(p0 * ip.yyyy) * ip.zzzz - 1.0;
    vec4 p0y = floor(fract(p0 * ip.yyyy) * 7.0) * ip.zzzz - 1.0;
    vec4 p0z = floor(fract(p0 * ip.zzzz) * 7.0) * ip.zzzz - 1.0;
    vec4 p1x = floor(p1 * ip.yyyy) * ip.zzzz - 1.0;
    vec4 p1y = floor(fract(p1 * ip.yyyy) * 7.0) * ip.zzzz - 1.0;
    vec4 p1z = floor(fract(p1 * ip.zzzz) * 7.0) * ip.zzzz - 1.0;

    vec4 norm0 = taylorInvSqrt(p0x * p0x + p0y * p0y + p0z * p0z);
    p0x *= norm0; p0y *= norm0; p0z *= norm0;
    vec4 norm1 = taylorInvSqrt(p1x * p1x + p1y * p1y + p1z * p1z);
    p1x *= norm1; p1y *= norm1; p1z *= norm1;

    vec3 grad0 = vec3(p0x.x, p0y.x, p0z.x);
    vec3 grad1 = vec3(p0x.y, p0y.y, p0z.y);
    vec3 grad2 = vec3(p0x.z, p0y.z, p0z.z);
    vec3 grad3 = vec3(p0x.w, p0y.w, p0z.w);
    vec3 grad4 = vec3(p1x.w, p1y.w, p1z.w);

    vec4 norm = max(0.6 - vec4(dot(x0.xyz, x0.xyz), dot(x1.xyz, x1.xyz), dot(x2.xyz, x2.xyz), dot(x3.xyz, x3.xyz)), 0.0);
    vec4 normSq = norm * norm;
    float n0 = normSq.x * normSq.x * dot(grad0, x0.xyz);
    float n1 = normSq.y * normSq.y * dot(grad1, x1.xyz);
    float n2 = normSq.z * normSq.z * dot(grad2, x2.xyz);
    float n3 = normSq.w * normSq.w * dot(grad3, x3.xyz);
    float norm4 = max(0.6 - dot(x4.xyz, x4.xyz), 0.0);
    float n4 = norm4 * norm4 * norm4 * norm4 * dot(grad4, x4.xyz);
    return 32.0 * (n0 + n1 + n2 + n3 + n4);
  }

  vec3 simplexVec3(vec3 p, float t) {
    return vec3(
      simplexNoise4d(vec4(p * 1.1, t * 0.11)),
      simplexNoise4d(vec4(p.yzx * 1.07 + 11.3, t * 0.13 + 3.1)),
      simplexNoise4d(vec4(p.zxy * 1.13 - 5.7, t * 0.17 + 8.3))
    );
  }

  vec3 curlNoise(vec3 p, float t) {
    float e = 0.075;
    vec3 dx = vec3(e, 0.0, 0.0);
    vec3 dy = vec3(0.0, e, 0.0);
    vec3 dz = vec3(0.0, 0.0, e);
    vec3 pX0 = simplexVec3(p - dx, t);
    vec3 pX1 = simplexVec3(p + dx, t);
    vec3 pY0 = simplexVec3(p - dy, t);
    vec3 pY1 = simplexVec3(p + dy, t);
    vec3 pZ0 = simplexVec3(p - dz, t);
    vec3 pZ1 = simplexVec3(p + dz, t);
    float x = (pY1.z - pY0.z) - (pZ1.y - pZ0.y);
    float y = (pZ1.x - pZ0.x) - (pX1.z - pX0.z);
    float z = (pX1.y - pX0.y) - (pY1.x - pY0.x);
    return normalize(vec3(x, y, z) / (2.0 * e + 1e-5));
  }

  float sdfSphere(vec3 p, float r) { return length(p) - r; }
  float sdfBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
  }
  float sdfTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
  }

  vec3 polarStream(vec3 p, float spin, float lift) {
    float r = length(p.xz) + 1e-4;
    float ang = atan(p.z, p.x);
    return vec3(-sin(ang), lift, cos(ang)) * spin * r;
  }

  vec3 foldSpace(vec3 p, float strength) {
    vec3 q = abs(p);
    q.xy = q.yx;
    q.yz = q.zy;
    return normalize(q - p) * strength;
  }

  vec3 evaluateField(vec3 p, vec3 velocity, float time, float progress, float pinch, float mode) {
    vec3 curl = curlNoise(p * (1.0 + progress * 1.6), time);
    vec3 noise = simplexVec3(p * (1.2 + pinch * 0.8), time * 0.7);
    float sph = sdfSphere(p, 0.55 + progress * 0.4);
    float tor = sdfTorus(p, vec2(0.4 + progress * 0.25, 0.12 + pinch * 0.08));
    float box = sdfBox(p, vec3(0.35 + progress * 0.2));
    vec3 radial = normalize(p + 1e-5);
    vec3 tangent = normalize(vec3(-p.z, 0.15 + pinch, p.x));
    vec3 field = curl;
    float m = floor(mode + 0.5);

    if (m < 0.5) field = -radial * (0.7 + progress) + curl * 0.8;
    else if (m < 1.5) field = curl * 1.5 + radial * sign(box) * 0.6;
    else if (m < 2.5) field = polarStream(p, 1.0 + pinch, 0.2 + progress * 0.3) + noise * 0.25;
    else if (m < 3.5) field = foldSpace(p, 1.0 + progress) + curl * 0.45;
    else if (m < 4.5) field = curl * 1.8 + tangent * 0.8;
    else if (m < 5.5) field = -radial * smoothstep(0.6, -0.2, sph) * 2.2 + noise * 0.2;
    else if (m < 6.5) field = curl * 0.8 + noise * 1.4 + vec3(0.0, 0.45, 0.0);
    else if (m < 7.5) field = vec3(-p.z, -tor, p.x) * 1.2 + curl * 0.35;
    else if (m < 8.5) field = tangent * (1.1 + progress) + vec3(0.0, -0.5, 0.0);
    else if (m < 9.5) field = vec3(sign(p.x), noise.y, sign(p.z)) * 0.9 + curl * 0.5;
    else if (m < 10.5) field = noise * 1.1 + vec3(sin(time + p.y * 4.0), cos(time + p.z * 3.0), sin(time + p.x * 2.0)) * 0.4;
    else if (m < 11.5) field = normalize(abs(noise) + 1e-4) * sign(noise) * 1.1 + radial * -0.4;
    else if (m < 12.5) field = curl * 0.7 + radial * sin(time + length(p) * 8.0) * 1.2;
    else if (m < 13.5) field = -tangent * (1.0 + pinch) - radial * 0.6;
    else if (m < 14.5) field = vec3(sin(time + p.y * 3.0), cos(time + p.z * 2.0), sin(time + p.x * 3.0)) + radial * -0.5;
    else if (m < 15.5) field = curl * 1.2 + noise * 0.9;
    else if (m < 16.5) field = radial * (1.4 + progress) + tangent * 0.7;
    else if (m < 17.5) field = vec3(noise.x, sign(sph) * 0.75, noise.z) + curl * 0.5;
    else if (m < 18.5) field = polarStream(p.yzx, 0.7 + progress, 0.4).zxy + noise * 0.4;
    else if (m < 19.5) field = foldSpace(p.zxy, 1.2).yzx + tangent * 0.8;
    else if (m < 20.5) field = sign(noise) * pow(abs(noise), vec3(2.0)) * 1.35 + radial * 0.4;
    else if (m < 21.5) field = curl * 1.7 + vec3(0.0, sin(time * 0.6) * 0.8, 0.0);
    else if (m < 22.5) field = normalize(velocity + noise + 1e-4) * 1.2;
    else if (m < 23.5) field = -radial / (0.25 + dot(p, p)) + curl * 0.6;
    else if (m < 24.5) field = vec3(-p.x, abs(p.y), p.z) * 0.9 + noise * 0.3;
    else if (m < 25.5) field = radial * (1.8 + pinch) + noise * 1.0;
    else if (m < 26.5) field = vec3(curl.xy, noise.z) * 1.25 + vec3(0.0, -progress, 0.0);
    else if (m < 27.5) field = polarStream(p, 1.4, 0.12) + radial * -0.8;
    else if (m < 28.5) field = normalize(vec3(noise.x + curl.x, curl.y, noise.z + curl.z) + 1e-4) * 1.7;
    else field = curl * 2.0 + velocity * 0.25 + radial * (progress - 0.4);

    field += smoothstep(0.25, -0.1, tor) * radial * 0.35;
    return field;
  }
`;
