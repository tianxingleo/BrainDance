# BrainDance 3DGS 移动端渲染方案

## 1. 架构概览

BrainDance 采用 **Flutter + WebView + WebGL** 混合架构在移动端实现 3D Gaussian Splatting (3DGS) 的实时渲染。核心思路是：Flutter 负责原生壳和资源管理，WebGL 负责实际的 3DGS 渲染计算。

```
┌─────────────────────────────────────────────────┐
│                  Flutter App                     │
│  ┌───────────────┐    ┌────────────────────────┐ │
│  │  原生 UI 层    │    │  webview_flutter       │ │
│  │  (Dart)       │───▶│  WebViewWidget         │ │
│  └───────────────┘    └──────────┬─────────────┘ │
│                                  │               │
│  ┌───────────────────────────────┼─────────────┐ │
│  │        本地 HttpServer        │             │ │
│  │  (127.0.0.1:random)          ▼             │ │
│  │  /index.html        → WebGL 查看器 HTML     │ │
│  │  /assets/*.js       → 渲染引擎 JS           │ │
│  │  /local_models/*    → 已下载的模型文件       │ │
│  │  /proxy/*           → 远程资源代理           │ │
│  └──────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              WebGL 渲染层 (WebView 内)            │
│                                                  │
│  ┌─────────────────┐  ┌──────────────────────┐  │
│  │  原版查看器       │  │  Spark 查看器         │  │
│  │  gaussian-       │  │  @sparkjsdev/spark   │  │
│  │  splats-3d      │  │  + 粒子矢量场         │  │
│  │  + Three.js     │  │  + 后处理管线         │  │
│  └─────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## 2. Flutter 端实现

### 2.1 入口页面：`WebGLViewerPage`

**文件位置**: `app/lib/pages/webgl_viewer.dart`

#### 核心职责

1. **平台检测与适配**：区分移动端 / 桌面端 / Web 平台
2. **模型下载与缓存**：支持断点续传、本地缓存
3. **本地 HTTP 服务**：在设备上起一个 `127.0.0.1` 服务器，为 WebView 提供资源
4. **WebView 管理**：初始化、通信、生命周期管理

#### 平台分发策略

| 平台 | 行为 |
|------|------|
| **Android / iOS** | 嵌入 WebView，在 App 内直接渲染 |
| **Windows / macOS / Linux** | 启动本地服务器 + 打开系统浏览器 |
| **Flutter Web** | 不支持，显示错误提示 |

### 2.2 本地 HTTP 服务器

Flutter 在 `initState` 阶段通过 `HttpServer.bind(InternetAddress.loopbackIPv4, 0)` 启动一个随机端口的本地服务器，处理四类请求：

| 路由 | 用途 |
|------|------|
| `/` 或 `/index.html` | 返回 WebGL 查看器的 HTML |
| `/assets/*` | 返回 JS/CSS 等静态资源（从 Flutter assets 读取） |
| `/local_models/<path>` | 返回已下载到设备本地的模型文件 |
| `/proxy/<encoded_url>` | 代理远程 HTTPS 请求，解决 WebView SSL 和 CORS 问题 |

所有响应均附加 CORS 头部（`Access-Control-Allow-Origin: *`、`Cross-Origin-Opener-Policy` 等），确保 WebGL 正常运行。

### 2.3 模型下载与缓存

```
远程 URL → 检查本地缓存 → 有 → 直接使用
                      → 无 → HTTP 下载（支持 Range 断点续传）
                           → 保存到 getApplicationDocumentsDirectory()
                           → 重命名 .tmp → 最终文件名
```

- 已下载模型缓存到 `getApplicationDocumentsDirectory()` 目录
- 下载过程中通过 `downloadEventBus` 广播进度
- 支持取消下载（`_downloadCancelled` 标志）

### 2.4 Flutter ↔ WebView 通信协议

**Flutter → WebView**: 通过 `runJavaScript` 调用 JS 全局函数

```dart
// 发送模型加载指令
_controller?.runJavaScript("window.loadModelFromFlutter($payload)");

// 发送 TimePeeling 模型列表
_controller?.runJavaScript(
  "window.setModelListForTimePeeling($json, '$currentModelId')"
);
```

**WebView → Flutter**: 通过 JavaScript Channel

```dart
_webViewController.addJavaScriptChannel(
  'BrainDanceChannel',
  onMessageReceived: (JavaScriptMessage message) {
    final data = jsonDecode(message.message);
    // 处理 ready / switchModel / error / info 等消息
  },
);
```

#### 消息格式

| 方向 | 事件 | Payload |
|------|------|---------|
| JS → Flutter | `status: 'ready'` | WebView 初始化完成 |
| JS → Flutter | `action: 'switchModel'` | `{ modelId: string }` 切换模型 |
| JS → Flutter | `status: 'error'` | `{ msg: string }` 错误信息 |
| Flutter → JS | `loadModelFromFlutter` | `{ ply, poses, matrix, imageId }` |
| Flutter → JS | `setModelListForTimePeeling` | 模型列表 + 当前模型 ID |

### 2.5 双查看器切换

Flutter 层支持在 "原版" 和 "Spark" 两个查看器之间切换，对应不同的本地资源目录：

```dart
String get _viewerAssetRoot =>
    _useSparkViewer ? 'assets/webgl_spark' : 'assets/webgl';
```

切换时重新加载 WebView 的 HTML 页面。

## 3. WebGL 渲染层

项目包含两套独立的 WebGL 查看器，均基于 Vue 3 + Vite 构建。

### 3.1 原版查看器（my-3dgs-viewer）

**目录**: `3dgs_viewer/my-3dgs-viewer/`

#### 技术栈

- **渲染引擎**: `@mkkellogg/gaussian-splats-3d` v0.4.7
- **3D 基础**: Three.js
- **动画**: GSAP
- **框架**: Vue 3 + Vite

#### 核心特性

- 支持 `.ply`、`.splat`、`.ksplat` 格式
- Orbit / Free 双相机模式
- 焦距（Focal Length）手动调节面板
- 基于标签的镜头搜索与飞行动画
- 电影模式（Cinematic）自动运镜
- TimePeeling 多模型切换

### 3.2 Spark 查看器（spark-3dgs-viewer）

**目录**: `3dgs_viewer/spark-3dgs-viewer/`

#### 技术栈

- **渲染引擎**: `@sparkjsdev/spark` v0.1.10（`SparkRenderer` + `SplatMesh` + `SparkControls`）
- **3D 基础**: Three.js
- **动画**: GSAP
- **后处理**: Three.js EffectComposer（Bloom + Afterimage）
- **框架**: Vue 3 + Vite

#### 渲染管线

```
                     ┌──────────────┐
                     │  SplatMesh   │  3DGS 高斯椭球渲染
                     │  (Spark)     │
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │ DataPipeline │  GPU 粒子矢量场（辅助特效）
                     │ (GLSL Shader)│
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │ SparkRenderer│  高斯排序与渲染调度
                     └──────┬───────┘
                            │
              ┌─────────────▼──────────────┐
              │     EffectComposer          │
              │  ┌────────────────────────┐ │
              │  │ 1. RenderPass          │ │
              │  │ 2. UnrealBloomPass     │ │  辉光效果
              │  │ 3. AfterimagePass      │ │  运动拖影
              │  └────────────────────────┘ │
              └─────────────────────────────┘
```

#### SparkGaussianViewer 组件流程

1. **初始化 Three.js 场景**：创建 Scene、PerspectiveCamera、WebGLRenderer
2. **创建 SparkRenderer**：配置高斯排序（`sortRadial: true`）
3. **创建 SparkControls**：相机交互控制
4. **加载 SplatMesh**：从 URL 加载 3DGS 模型
5. **等待模型初始化**：`await splatMesh.initialized`
6. **计算场景边界**：从 BoundingBox 提取中心和半径
7. **挂载特效**：球体高亮（SplatEditSdf）+ 剖切平面
8. **加载位姿数据**：从 JSON 获取相机位姿列表
9. **启动渲染循环**：`renderer.setAnimationLoop()`

### 3.3 BrainDance Engine（原版查看器的引擎层）

**文件**: `3dgs_viewer/spark-3dgs-viewer/src/engine/brainDance-engine.js`

这是原版查看器的底层渲染引擎（不同于 SparkGaussianViewer），采用面向对象的类设计：

```javascript
class BrainDanceEngine {
  constructor(mount) {
    // 初始化 Three.js 场景、相机、渲染器
    // 初始化 SparkRenderer + SparkControls
    // 初始化 EffectComposer（Bloom + Afterimage）
    // 初始化 DataPipeline（GPU 粒子模拟）
    // 初始化 CinematographyEngine（运镜引擎）
    // 初始化 PerformanceIO（UI 交互）
  }
}
```

#### 特殊功能

- **CinematographyEngine**: 自动运镜，支持序列播放、焦点追踪
- **DataPipeline**: GPU 加速粒子矢量场，用 GLSL Shader 在 RenderTarget 中做物理模拟
- **PerformanceIO**: UI 控制面板的 I/O 管理

## 4. 移动端触摸交互

### 4.1 触摸检测与手势分类

引擎在初始化时通过以下方式判断是否为移动端：

```javascript
this.isMobile =
  window.matchMedia('(pointer: coarse)').matches ||
  'ontouchstart' in window ||
  (navigator.maxTouchPoints || 0) > 0 ||
  window.innerWidth <= 820;
```

### 4.2 自定义触摸控制系统

移动端禁用 SparkControls 的 pointerControls，改用完全自定义的触摸处理：

#### 三种手势

| 手势 | 触发条件 | 行为 |
|------|----------|------|
| **单指拖动** | `touches.length === 1` | Orbit 旋转（绕焦点旋转相机） |
| **双指捏合** | `touches.length >= 2` + 距离变化 | Zoom 缩放 |
| **双指平移** | `touches.length >= 2` + 中点移动 | Pan 平移 |

#### 移动端交互参数

```javascript
const MOBILE_ROTATE_SPEED = 0.0055;  // 旋转灵敏度
const MOBILE_PAN_SPEED = 0.0016;     // 平移灵敏度
const MOBILE_ZOOM_SPEED = 1.0;       // 缩放灵敏度
```

#### Orbit 实现原理

```javascript
_mobileOrbit(dx, dy, focus) {
  // 1. 计算相机相对于焦点的偏移向量
  // 2. 转换为球坐标 (Spherical)
  // 3. 根据 dx/dy 修改 theta/phi
  // 4. phi 限制在 [0.18, π-0.18] 防止极点翻转
  // 5. 重新计算相机位置
  // 6. 关闭自动运镜
}
```

#### UI 手势隔离

当用户正在操作 UI 控件（按钮、滑块、搜索框等）时，触摸事件不会传递到 3D 渲染层：

```javascript
const isUiTarget = (target) =>
  target.closest('.bd-topbar, .bd-console, .bd-help, .bd-pose-dock, .bd-focus-card, .bd-status');
```

### 4.3 Spark 查看器的交互参数

SparkGaussianViewer 中的交互参数（非移动端自适应）：

```javascript
controls.pointerControls.rotateSpeed = 0.0018;
controls.pointerControls.slideSpeed = 0.0045;
controls.pointerControls.scrollSpeed = 0.0013;
```

BrainDance Engine 中根据屏幕宽度自适应：

```javascript
// 移动端使用更大的灵敏度补偿触摸精度
this.controls.pointerControls.rotateSpeed = window.innerWidth <= 820 ? 0.0022 : 0.0015;
this.controls.pointerControls.slideSpeed = window.innerWidth <= 820 ? 0.0062 : 0.0042;
this.controls.pointerControls.scrollSpeed = window.innerWidth <= 820 ? 0.0010 : 0.0014;
this.controls.pointerControls.reverseSwipe = true;  // 反向滑动
```

## 5. 数据管线（DataPipeline）

**文件**: `3dgs_viewer/spark-3dgs-viewer/src/engine/data-pipeline.js`

### 5.1 功能定位

DataPipeline 是一个 **GPU 加速的粒子矢量场系统**，作为 3DGS 的辅助视觉特效叠层。它不是核心的 3DGS 渲染器，而是提供额外的粒子动画效果。

### 5.2 工作流程

```
PLY 文件 → 解析顶点位置/颜色 → 归一化 → 写入 DataTexture
                                                  │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │  GPU Ping-Pong 模拟     │
                                    │  RenderTarget A ↔ B     │
                                    │                         │
                                    │  velocityMaterial:      │
                                    │    计算矢量场加速度      │
                                    │    更新速度              │
                                    │                         │
                                    │  positionMaterial:      │
                                    │    根据速度更新位置      │
                                    │    添加位置约束 (leash)  │
                                    └─────────┬───────────────┘
                                              │
                                              ▼
                                    THREE.Points (ShaderMaterial)
                                    自定义顶点/片元着色器渲染
```

### 5.3 PLY 解析

支持 ASCII 和 Binary 两种 PLY 格式：

```javascript
// 自动检测头结束位置
function detectHeaderEnd(bytes) { ... }

// 解析头部元信息（format、vertexCount、properties）
function parsePlyHeader(text) { ... }

// 分别处理 ASCII 和 Binary 顶点数据
function parseAsciiVertices(bodyText, header) { ... }
function parseBinaryVertices(buffer, dataOffset, header) { ... }
```

### 5.4 粒子渲染着色器

- **顶点着色器**: 从 DataTexture 读取位置，计算深度衰减和脉冲大小
- **片元着色器**: 圆形粒子遮罩 + 颜色混合 + 模式着色

## 6. 相机系统与位姿管理

### 6.1 位姿文件格式

位姿数据以 JSON 格式存储，结构如下：

```json
{
  "w": 1920,
  "h": 1080,
  "fl_x": 800,
  "fl_y": 800,
  "frames": [
    {
      "id": "image_001",
      "matrix": [4x4 变换矩阵, 列主序, 16 个 float],
      "image_url": "path/to/image.jpg",
      "tag": "正面全景",
      "fl_x": 800,
      "fl_y": 800
    }
  ]
}
```

### 6.2 飞行到指定视角（flyToImage）

```javascript
flyToImage(poseData) {
  // 1. 从 matrix 解析出 position + quaternion
  // 2. 使用 GSAP 动画平滑过渡相机位置和朝向
  //    - position: 0.9s, power2.inOut
  //    - quaternion: 0.9s, power2.inOut
  // 3. 同时动画过渡焦距（0.65s）
  // 4. 更新高亮效果到当前关注区域
}
```

### 6.3 焦距管理

支持在相机的 FOV 和实际焦距（像素）之间换算：

```javascript
// FOV → 焦距
const focal = (h * 0.5) / Math.tan(fov * Math.PI / 360);

// 焦距 → FOV
const fov = 2 * Math.atan(h * 0.5 / focalPx) * 180 / Math.PI;
```

### 6.4 CinematographyEngine

自动运镜引擎，支持：
- 自动环绕场景
- 焦点追踪和插值
- 序列运镜播放
- 手动控制时自动暂停

## 7. 特效系统

**文件**: `3dgs_viewer/spark-3dgs-viewer/src/lib/sparkEffects.js`

### 7.1 球体高亮（Sphere Highlight）

```javascript
// 使用 Spark 的 SplatEdit + SplatEditSdf
const edit = new SplatEdit({
  rgbaBlendMode: SplatEditRgbaBlendMode.ADD_RGBA,  // 加法混合
  softEdge: sceneRadius * 0.18,                      // 柔和边缘
});

const sdf = new SplatEditSdf({
  type: SplatEditSdfType.SPHERE,
  color: new THREE.Color('#d86f3d'),  // 橙色高亮
  opacity: 0.55,
  radius: sceneRadius * 0.16,
});
```

### 7.2 剖切平面（Clip Plane）

```javascript
// 使用乘法混合 + 反转实现遮挡效果
const edit = new SplatEdit({
  rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY,
  invert: true,
});
```

支持动态调整剖切位置（clipOffset 参数），实时预览模型内部结构。

## 8. 模型格式支持

| 格式 | 扩展名 | 说明 | 渲染方式 |
|------|--------|------|----------|
| PLY | `.ply` | 点云格式，支持 ASCII/Binary | SplatMesh + DataPipeline 粒子 |
| Splat | `.splat` | 3DGS 专用格式 | SplatMesh（跳过 PLY 解析） |
| KSplat | `.ksplat` | Kosmos 格式 | SplatMesh（跳过 PLY 解析） |

## 9. 后端训练管线

**目录**: `ai_engine/3dgs/`

### 9.1 训练流程

- 使用 **Nerfstudio + Splatfacto** 进行 3DGS 训练
- 集成 **ml-sharp** (Sharp) 模型进行单图到 3DGS 的推理
- 集成 **SAM-3D-Objects** 进行 3D 目标分割
- 通过 **Supabase** 进行云端任务队列管理

### 9.2 输出产物

训练完成后产出：
- `point_cloud.ply` / `.splat` / `.ksplat` — 3DGS 模型文件
- `webgl_poses.json` / `webgl_poses_with_tags.json` — 相机位姿文件
- 预览图（thumbnail）

这些产物上传到 Supabase Storage (`braindance-assets` bucket)，Flutter 通过公开 URL 下载后在本地渲染。

## 10. TimePeeling（多模型时间线）

### 10.1 概念

TimePeeling 允许用户在同一个场景的多个不同版本的 3DGS 模型之间切换浏览（如不同时间拍摄的场景）。

### 10.2 实现流程

```
Flutter 构建模型列表 → setModelListForTimePeeling() → JS 端显示底部卡片
                                                                │
用户点击卡片 → JS 发送 switchModel → Flutter 查找目标模型 → 下载/缓存
                                                                │
                                                                ▼
                                            loadModelFromFlutter() → 切换渲染
```

## 11. 移动端性能优化措施

| 优化项 | 实现 |
|--------|------|
| **像素比限制** | `setPixelRatio(Math.min(devicePixelRatio, 2))` |
| **高性能 GPU** | `powerPreference: 'high-performance'` |
| **抗锯齿关闭** | `antialias: false` |
| **触摸事件阻止默认** | `touch-action: none` + `event.preventDefault()` |
| **viewport 锁定** | `maximum-scale=1.0, user-scalable=no` |
| **UI 手势隔离** | 操作 UI 控件时屏蔽 3D 渲染层触摸 |
| **模型本地缓存** | 避免重复下载大模型文件 |
| **断点续传** | HTTP Range 请求支持 |
| **本地代理** | 避免 WebView 的 SSL/CORS 限制 |
| **自适应灵敏度** | 移动端使用不同的交互速度参数 |

## 12. 文件结构索引

```
app/
├── lib/pages/webgl_viewer.dart              # Flutter 端 WebView 查看器页面
├── assets/
│   ├── webgl/                                # 原版查看器构建产物
│   │   ├── index.html
│   │   └── assets/index-q4qg3mPN.js
│   └── webgl_spark/                          # Spark 查看器构建产物
│       ├── index.html
│       └── assets/index-CH-kkXe7.js

3dgs_viewer/
├── my-3dgs-viewer/                           # 原版查看器源码
│   ├── src/
│   │   ├── components/GaussianViewer.vue     # 核心 3DGS 查看器组件
│   │   └── components/BottomSelector.vue     # 底部镜头选择器
│   └── package.json                          # 依赖: gaussian-splats-3d
└── spark-3dgs-viewer/                        # Spark 查看器源码
    ├── src/
    │   ├── components/SparkGaussianViewer.vue # Spark 3DGS 查看器
    │   ├── engine/brainDance-engine.js        # 底层渲染引擎
    │   ├── engine/data-pipeline.js            # GPU 粒子矢量场
    │   ├── engine/cinematography-io.js        # 运镜引擎
    │   ├── engine/vector-fields.js            # 矢量场定义
    │   ├── lib/viewerBridge.js                # Flutter 通信桥
    │   ├── lib/sparkEffects.js                # 高亮/剖切特效
    │   ├── lib/cameraMath.js                  # 焦距计算
    │   └── lib/poseUtils.js                   # 位姿工具
    └── package.json                           # 依赖: @sparkjsdev/spark

ai_engine/3dgs/                                # 后端训练管线
└── src/
    ├── pipelines/image_to_3d.py               # 图像到 3DGS 流水线
    ├── pipelines/mask_guided.py               # 掩码引导流水线
    └── libs/
        ├── ml-sharp/                          # Sharp: 单图 3DGS 推理
        └── sam-3d-objects/                    # SAM-3D: 3D 目标分割
```

## 13. 电影模式（Cinematic Mode）算法详解

项目中存在两套不同的电影运镜系统，分别用于两个查看器。

---

### 13.1 Spark 查看器的自动环绕运镜（CinematographyEngine）

**文件**: `3dgs_viewer/spark-3dgs-viewer/src/engine/cinematography-io.js`

#### 核心算法：三次贝塞尔曲线路径

相机沿着一条**闭合三次贝塞尔曲线**自动环绕场景运动。

```
数学公式:
B(t) = (1-t)³·P0 + 3(1-t)²·t·P1 + 3(1-t)·t²·P2 + t³·P3

其中:
- t ∈ [0, 1]，一个周期 12 秒
- P0, P1, P2 是预设偏移量，P3 = P0 形成闭合环
- 最终相机位置 = sceneCenter + basisTransform(B(t))
```

#### 预设路径序列

系统预计算了 **15 条不同的运动序列**，每条序列包含 3 个 3D 控制点（乘以 `sceneRadius` 缩放）：

```javascript
// 示例：第 1 条序列
[
  new THREE.Vector3(0.0, 0.15, 1.4),    // P0: 正前方略高
  new THREE.Vector3(0.8, 0.25, 1.1),    // P1: 右前方
  new THREE.Vector3(1.2, -0.1, 0.5),    // P2: 右侧偏低
]
// P3 = P0 → 形成闭合曲线

// 示例：第 6 条序列 — 高位俯瞰
[
  new THREE.Vector3(-0.6, 0.9, 1.2),    // 左前上方
  new THREE.Vector3(0.3, 1.3, 0.7),     // 正上方
  new THREE.Vector3(1.0, 0.7, -0.4),    // 右后上方
]
```

#### 自适应坐标系（Basis Transform）

贝塞尔曲线的坐标不是世界坐标，而是相对于当前相机观察方向的**局部坐标系**：

```javascript
_basis() {
  // 1. 计算当前观察方向
  const currentDir = camera.position - focusLerp → normalize

  // 2. 用指数移动平均平滑观察方向（系数 0.18）
  userIntent.lerp(currentDir, 0.18) → normalize

  // 3. 构建 right/up/forward 正交基
  right   = cross(baseUp, userIntent) → normalize
  up      = cross(userIntent, right)  → normalize
  forward = userIntent
}
```

这意味着**路径会跟随用户的观察方向旋转**，不会突然切换到完全不相关的角度。

#### 位置平滑（指数衰减插值）

```javascript
// 每帧更新
focusLerp.lerp(focus, 1 - exp(-dt * 4.0))    // 焦点平滑追踪
camera.position.lerp(targetPos, 1 - exp(-dt * 2.8))  // 相机位置平滑
camera.lookAt(focusLerp)                       // 始终朝向焦点
```

这种 `1 - exp(-dt * k)` 的方式是帧率无关的指数衰减插值，比简单的 `lerp(a, b, alpha)` 更稳定。

#### 飞行到指定视角（flyToPose）

```javascript
flyToPose(cameraState) {
  // 1. 从位姿 matrix 解析出目标 position + quaternion
  // 2. 计算焦点: position + forward * sceneRadius * 0.8
  // 3. 使用 GSAP 动画:
  //    - duration: 1.2s
  //    - ease: power2.inOut
  //    - position: 线性插值 lerpVectors
  //    - quaternion: 球面插值 slerpQuaternions
  //    - focus: 线性插值
}
```

---

### 13.2 原版查看器的智能电影运镜

**文件**: `3dgs_viewer/my-3dgs-viewer/src/components/GaussianViewer.vue`

这是一个更复杂、更智能的电影模式系统，包含完整的**关键帧选取 → 路径规划 → 曲线插值 → 阻尼跟踪**流程。

#### 步骤 1：关键帧选取（selectStableCinematicKeyframes）

从所有相机位姿中筛选出最多 18 个（最少 6 个）最佳关键帧：

```javascript
// 评分公式
score = upAlignment * 2.2               // 相机是否水平（不倾斜）
      + directionalContinuity * 1.4     // 前后帧方向是否一致
      + Math.min(avgDistance, 1.5) * 0.4 // 空间分布均匀性

// 过滤条件
- 保留首尾帧（forced）
- upAlignment >= 0.45（排除过于倾斜的视角）
- 按 score 降序选取
```

#### 步骤 2：智能路径规划（planSmartCinematicRoute）

自动分析场景空间结构，选择最适合的运镜方式：

```javascript
// 分析参数
angleSpread    → 所有关键帧相对于场景中心的角度分布范围
radiiStd       → 关键帧到中心距离的标准差
heightSpread   → 关键帧高度差

// 三种运镜模式
if (angleSpread > 1.1 且 radiiStd < mean*0.28)  → 'orbit' 环绕模式
   // 按角度排序，形成环绕路径

else if (heightSpread > max(0.8, mean*0.42))     → 'crane' 摇臂模式
   // 按高度排序，形成上升/下降路径

else                                               → 'dolly' 推拉模式
   // 按主水平轴投影排序，形成直线推拉路径
```

#### 步骤 3：路径方向优化（chooseLowerCostRouteDirection）

正向和反向都计算一遍转移代价，选择代价更低的方向：

```javascript
transitionCost = distance * 1.25              // 空间距离
               + forwardMismatch * 1.4        // 观察方向突变
               + focusMismatch * 0.9           // 焦点方向突变
               + heightDelta * 0.35            // 高度跳变
```

#### 步骤 4：平滑处理

多种平滑算法串联使用：

1. **四元数平滑** (`smoothQuaternionSeries`)
   - 先保证四元数连续性（处理符号翻转）
   - 多次迭代球面线性插值 (slerp) 低通滤波
   - `passes = 1 + round(smoothness * 3)`, `blend = 0.16 + smoothness * 0.22`

2. **位置平滑** (`smoothVectorSeries`)
   - 多次迭代加权平均：`blended = (prev + curr*2 + next) / 4`
   - 然后与原始值 lerp 混合

3. **焦距平滑** (`smoothScalarSeries`)
   - 与位置平滑相同的算法，应用于焦距标量值

#### 步骤 5：CatmullRom 曲线插值

```javascript
// 位置曲线 — CatmullRom (centripetal 参数化)
const curve = new THREE.CatmullRomCurve3(positions, false, 'centripetal');

// 注视目标曲线 — 同样是 CatmullRom
const lookCurve = new THREE.CatmullRomCurve3(targets, false, 'centripetal');
```

Centripetal 参数化比 uniform 和 chordal 更不容易产生尖角和交叉。

#### 步骤 6：沿曲线采样（sampleCinematicTrajectory）

```javascript
// 1. 按路径累积距离均匀采样（不是按参数 t 均匀）
// 2. 位置: curve.getPointAt(t)  — 等弧长参数化
// 3. 朝向: 前后关键帧四元数 slerp
// 4. 注视: 前后目标点 lerp
// 5. 焦距: 前后焦距 lerp
// 6. 使用 smootherstep 做缓动
```

#### 步骤 7：阻尼跟踪（applyCinematicSample）

实际相机不是直接跳到采样点，而是通过阻尼追踪：

```javascript
// 阻尼系数由 cinematicSmoothness 参数控制
dampingAlpha = lerp(0.26, 0.10, smoothness)  // 快 → 慢

// 位置阻尼
filteredPosition.lerp(samplePosition, dampingAlpha)

// 旋转阻尼
filteredQuaternion.slerp(sampleQuaternion, dampingAlpha)

// 焦距阻尼
filteredFocal = lerp(current, target, dampingAlpha * 0.85)
```

#### 步骤 8：循环桥接（buildLoopBridgeSegment）

运镜结束时，构建一条空中过渡路径回到起点：

```javascript
// 从最后一帧 → 抬高 + 外推 → 空中过渡 → 降低 → 回到第一帧
bridgePositions = [
  last.position,
  last.position + (0, lift, 0) + radialPush,     // 抬高外推
  first.position + (0, lift*0.86, 0) + radialPush, // 接近起点
  first.position,                                   // 回到起点
]

// 注视目标过渡到场景中心
bridgeTargets = [
  last.target → lerp(centerLift, 0.4),
  centerLift,                                        // 中心
  centerLift,                                        // 中心
  first.target → lerp(centerLift, 0.28),
]
```

#### 持续时间计算

```javascript
durationMs = clamp(
  totalDistance * 1600 + segmentCount * 260,  // 距离越远 + 关键帧越多 → 越长
  7000,   // 最短 7 秒
  42000   // 最长 42 秒
) / cinematicSpeed  // 速度倍率
```

---

### 13.3 MediaPipe 手势控制（PerformanceIO）

Spark 查看器还集成了 **MediaPipe HandLandmarker** 做手势识别：

```javascript
// 检测拇指和食指的捏合距离
const wrist  = landmarks[0];
const thumb  = landmarks[4];
const index  = landmarks[8];
const middle = landmarks[9];

const palmSize = distance(wrist, middle);          // 手掌大小（归一化基准）
const pinchRaw = distance(thumb, index) / palmSize; // 捏合比
const pinch    = clamp(1.0 - pinchRaw, 0, 1);      // 捏合程度 0~1

// pinch 值驱动粒子矢量场的 pinch 参数和视觉预览
```

---

## 14. 移动端触摸手势系统详解

### 14.1 触摸信号检测

是的，移动端手势**完全基于浏览器原生 TouchEvent 信号**。系统不使用任何第三方手势库，而是直接监听底层触摸事件：

```javascript
surface.addEventListener('touchstart', onTouchStart, { passive: false });
surface.addEventListener('touchmove', onTouchMove, { passive: false });
surface.addEventListener('touchend', onTouchEnd, { passive: false });
surface.addEventListener('touchcancel', onTouchEnd, { passive: false });
```

同时阻止了 canvas 上的默认浏览器手势行为：

```javascript
['touchstart', 'touchmove', 'gesturestart', 'gesturechange', 'gestureend'].forEach(
  (eventName) => {
    canvas.addEventListener(eventName, (event) => event.preventDefault(), { passive: false });
  }
);
// CSS: touch-action: none; user-select: none; -webkit-touch-callout: none;
```

### 14.2 触摸状态机

系统维护一个 `touchState` 对象追踪触摸状态：

```javascript
this.touchState = {
  dragging: false,      // 单指拖动中
  pinching: false,      // 双指操作中
  lastX: 0,             // 上一次单指 X 坐标
  lastY: 0,             // 上一次单指 Y 坐标
  lastDistance: 0,       // 上一次双指间距
  lastMidX: 0,          // 上一次双指中点 X
  lastMidY: 0,          // 上一次双指中点 Y
};
```

### 14.3 三种手势的判定逻辑

#### touchstart 事件处理

```
if (touches.length >= 2) {
  → 进入双指模式
  → pinching = true, dragging = false
  → 记录两指距离和中点
}

else if (touches.length === 1) {
  → 进入单指模式
  → dragging = true, pinching = false
  → 记录触点坐标
}
```

#### touchmove 事件处理

```
if (touches.length >= 2) {
  → 双指操作
  → 计算新距离和中点

  if (距离变化) → _mobileZoom(scale, focus)
     scale = newDistance / lastDistance
     算法: 球坐标调整相机到焦点的距离
     公式: newDistance = clamp(currentDistance / scale, min, max)
     限制: [sceneRadius * 0.6, sceneRadius * 6.0]

  if (中点移动) → _mobilePan(dx, dy)
     算法: 在相机的 right/up 平面上平移
     公式: offset = right * (-dx * PAN_SPEED * distance) + up * (dy * PAN_SPEED * distance)
     同时平移 camera.position 和 focus
}

else if (touches.length === 1 && dragging) {
  → 单指操作
  → _mobileOrbit(dx, dy, focus)

  算法:
  1. offset = camera.position - focus
  2. 转球坐标: Spherical(theta, phi, radius)
  3. theta -= dx * ROTATE_SPEED   (水平旋转)
  4. phi   += dy * ROTATE_SPEED   (垂直旋转)
  5. phi 限制在 [0.18, π-0.18]    (防止极点翻转)
  6. camera.position = focus + offset.setFromSpherical()
  7. camera.lookAt(focus)
}
```

#### touchend 事件处理

```
if (touches >= 2) → 更新为剩余两指的距离和中点
if (touches == 1) → 切换为单指追踪
if (touches == 0) → 重置所有状态
```

### 14.4 UI 手势隔离

当用户在 UI 控件上操作时，触摸事件不会传递到 3D 渲染层：

```javascript
// 判断触摸目标是否是 UI 元素
const isUiTarget = (target) =>
  target.closest('.bd-topbar, .bd-console, .bd-help, .bd-pose-dock, .bd-focus-card, .bd-status');

// UI 操作时
_setUiInteraction(true) {
  canvasLayer.style.pointerEvents = 'none'     // 禁用 canvas 事件
  controls.pointerControls.enable = false       // 禁用控件
}

// UI 操作结束时
_setUiInteraction(false) {
  canvasLayer.style.pointerEvents = 'auto'      // 恢复 canvas 事件
  if (!isMobile) controls.pointerControls.enable = true  // 桌面端恢复控件
}
```

同时通过事件冒泡阻止实现双重保障：

```javascript
// UI 元素上拦截触摸事件
['button', 'input', 'select', 'label', '.bd-console', ...].forEach(selector => {
  element.addEventListener('touchstart', (e) => { e.stopPropagation(); e.stopImmediatePropagation(); });
  element.addEventListener('touchmove', (e) => { e.stopPropagation(); e.stopImmediatePropagation(); });
});

// 触摸开始时标记 UI 交互状态
element.addEventListener('touchstart', () => { startUiInteraction(); });
element.addEventListener('touchend', () => { endUiInteraction(); });
```

### 14.5 完整手势流程图

```
touchstart
    │
    ├── touches >= 2 ?
    │       ├── YES → pinching=true, dragging=false
    │       │         记录 lastDistance, lastMidX, lastMidY
    │       │
    │       └── NO (touches == 1)
    │               ├── isUiTarget? → 忽略，不处理
    │               └── 否 → dragging=true, pinching=false
    │                        记录 lastX, lastY
    │
touchmove
    │
    ├── uiInteracting? → 忽略
    │
    ├── touches >= 2 ?
    │       ├── 距离变化 → _mobileZoom(scale)
    │       │    球坐标系调整距离
    │       │
    │       └── 中点移动 → _mobilePan(dx, dy)
    │            相机 right/up 平面平移
    │
    ├── touches == 1 && dragging?
    │       └── _mobileOrbit(dx, dy)
    │            球坐标系 theta/phi 旋转
    │            phi 限制 [0.18, π-0.18]
    │
    └── 所有操作 → 关闭 autoPlay/autoCamera
                    相机进入手动控制模式

touchend
    │
    ├── touches >= 2 → 更新为剩余手指状态
    ├── touches == 1 → 切换为单指追踪
    └── touches == 0 → 重置 dragging/pinching/lastDistance
```

---

## 16. 架构优缺点分析

### 优点

1. **跨平台一致性**：同一套 WebGL 代码在 Android/iOS/桌面浏览器中运行
2. **渲染引擎可插拔**：原版和 Spark 查看器可无缝切换
3. **热更新友好**：WebGL 部分可以独立更新，无需发布 App 新版本
4. **丰富的交互**：支持 Orbit/Pan/Zoom + 镜头飞行 + 自动运镜
5. **模型格式兼容性好**：支持 PLY/Splat/KSplat 三种主流格式

### 局限

1. **性能瓶颈**：WebGL 相比原生 OpenGL/Vulkan/Metal 有性能开销
2. **无法利用原生 GPU 特性**：无法使用 ARKit/ARCore、Metal 性能着色器等平台特性
3. **内存管理受限**：WebView 的内存管理不如原生应用精细
4. **大模型加载慢**：百 MB 级别的 3DGS 模型在移动端 WebView 中加载时间较长
5. **电池与发热**：持续 WebGL 渲染对移动设备电池和温度有影响
