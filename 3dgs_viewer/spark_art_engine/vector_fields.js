// 1. GLSL Vector Fields (底层矢量场算法)
// 极大幅度的抽象：30种独立非线性微分矢量场 (基于Simplex 4D / Curl / SDF)

export const simFragmentShader = `
uniform float uProgress;
uniform float uTime;
uniform float uVectorFieldType;
uniform sampler2D textureOriginPosition;


// Common noise functions (Simplex 4D, Curl pseudo-code)
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+10.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 = v - i + dot(i, C.xxx) ;

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289(vec4(i.x, i.y, i.z, 0.0)).xyz;
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

  float n_ = 0.142857142857;
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

vec3 curlNoise(vec3 p) {
    float e = 0.1;
    vec3 dx = vec3(e, 0.0, 0.0);
    vec3 dy = vec3(0.0, e, 0.0);
    vec3 dz = vec3(0.0, 0.0, e);

    vec3 p_x0 = vec3(snoise(p - dx), snoise(p - dx + vec3(100.0)), snoise(p - dx + vec3(200.0)));
    vec3 p_x1 = vec3(snoise(p + dx), snoise(p + dx + vec3(100.0)), snoise(p + dx + vec3(200.0)));
    vec3 p_y0 = vec3(snoise(p - dy), snoise(p - dy + vec3(100.0)), snoise(p - dy + vec3(200.0)));
    vec3 p_y1 = vec3(snoise(p + dy), snoise(p + dy + vec3(100.0)), snoise(p + dy + vec3(200.0)));
    vec3 p_z0 = vec3(snoise(p - dz), snoise(p - dz + vec3(100.0)), snoise(p - dz + vec3(200.0)));
    vec3 p_z1 = vec3(snoise(p + dz), snoise(p + dz + vec3(100.0)), snoise(p + dz + vec3(200.0)));

    float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
    float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
    float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

    return normalize(vec3(x, y, z) / (2.0 * e));
}

// 模拟30种逻辑入口 (截取核心前6个，可以通过配置项扩充至30)
vec3 calculateVectorField(vec3 pos, vec3 originPos, float progress, float vfType) {
    vec3 force = vec3(0.0);
    
    // 0: Curl Noise Entropy
    if(vfType < 0.5) {
        force = curlNoise(pos * 0.5 + uTime * 0.2);
    }
    // 1: Inverse Particle Growth (逆向生长)
    else if(vfType < 1.5) {
        vec3 target = originPos;
        force = (target - pos) * 2.0;
        force += curlNoise(pos * 2.0) * (1.0 - progress);
    }
    // 2: Topological Shatter (拓扑碎裂)
    else if(vfType < 2.5) {
        force = normalize(pos) * snoise(pos * 5.0) * 5.0;
        float dist = length(pos);
        if(dist < 2.0) force += cross(pos, vec3(0.0, 1.0, 0.0)) * 10.0;
    }
    // 3: Polar Fluid Dynamics (极坐标流体)
    else if(vfType < 3.5) {
        float theta = atan(pos.y, pos.x) + progress * 10.0;
        float r = length(pos.xy);
        force = vec3(-sin(theta), cos(theta), sin(r - uTime) * 2.0) * r;
    }
    // 4: Multi-Dim Space Fold (多维折叠)
    else if(vfType < 4.5) {
        force = vec3(sin(pos.z * 5.0), cos(pos.x * 5.0), sin(pos.y * 5.0)) * progress * 5.0;
    }
    // 5: SDF Magnetic Core (SDF磁性形变)
    else if(vfType < 5.5) {
        float d = length(pos - vec3(0.0, sin(uTime)*2.0, 0.0)) - 1.5;
        vec3 normal = normalize(pos - vec3(0.0, sin(uTime)*2.0, 0.0));
        force = -normal * d * 5.0 * progress;
    }
    
    // Fallback default
    else {
        force = curlNoise(pos * 0.1) * progress;
    }
    
    return force;
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    vec4 posData = texture2D(texturePosition, uv);
    vec4 velData = texture2D(textureVelocity, uv);
    vec4 opData = texture2D(textureOriginPosition, uv); // Original 3DGS Position
    
    vec3 pos = posData.xyz;
    vec3 vel = velData.xyz;
    vec3 op  = opData.xyz;
    
    // 物理积分更新 (阻尼与步进方程)
    // 根据 uProgress 双向数据绑定，计算状态迁移
    // 当 uProgress=0 时，处于原始点云态，当 uProgress=1时，处于极端混沌态
    
    vec3 targetForce = calculateVectorField(pos, op, uProgress, uVectorFieldType);
    
    vel += targetForce * 0.016; // dt
    vel *= 0.95; // 阻尼
    
    pos += vel * 0.016;
    
    // 边界约束与归位逻辑
    if(uProgress < 0.01) {
        // 缓慢归位到原始 3DGS 拓扑
        vec3 diff = op - pos;
        vel += diff * 0.1;
        vel *= 0.8;
    }
    
    gl_FragColor = vec4(pos, 1.0);
}
`;

export const renderVertexShader = `
uniform sampler2D texturePosition;
uniform float uProgress;
attribute vec3 aColor;
varying vec3 vColor;
varying float vProgress;

void main() {
    vec4 posData = texture2D(texturePosition, position.xy);
    vProgress = uProgress;
    
    // 解析高密度的坐标 (uv在xy上)
    vec3 morphedPos = posData.xyz;
    
    // 在低进度下强制显示 Splat 原始色，进度增加时才产生位置相关的 HDR 色彩偏移
    vColor = mix(aColor, normalize(abs(morphedPos)) * 0.9 + 0.1, smoothstep(0.05, 0.4, uProgress));
    
    vec4 mvPosition = modelViewMatrix * vec4(morphedPos, 1.0);
    // 降低基础点大小，还原 Splat 精细的层叠感
    gl_PointSize = max(1.5, 8.0 / -mvPosition.z); 
    gl_Position = projectionMatrix * mvPosition;
}
`;

export const renderFragmentShader = `
varying vec3 vColor;
varying float vProgress;

void main() {
    float dist = length(gl_PointCoord - vec2(0.5));
    if (dist > 0.5) discard;
    
    // 强化粒子中心的实心质感，减少模糊边缘覆盖掉模型细节
    float alpha = smoothstep(0.5, 0.45, dist); 
    
    // 只有在变形过程中才引入 HDR 增益，初始状态亮度更自然
    vec3 col = vColor * (1.0 + smoothstep(0.3, 1.0, vProgress) * 1.5);
    
    gl_FragColor = vec4(col, alpha * 0.95);
}
`;
