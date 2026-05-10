# BrainDance VR 3DGS Viewer

这是 BrainDance 的独立 VR 预览端，定位是 PC WebXR / SteamVR 查看器。PICO Neo 2 这类旧安卓头显不需要直接安装新 App，只作为 PCVR 串流显示端使用。

## 运行方式

```bash
cd 3dgs_viewer/vr-3dgs-viewer
npm install
npm run dev
```

浏览器打开：

```text
https://127.0.0.1:5174/?preview=desktop
```

WebXR 需要安全上下文，本端通过 Vite basic SSL 在开发环境提供 HTTPS。

## 预览模式

VR 端支持三种调试模式：

```text
https://127.0.0.1:5174/?preview=desktop
https://127.0.0.1:5174/?preview=stereo
https://127.0.0.1:5174/?preview=webxr
```

- `desktop`：普通桌面预览，使用同一套 payload / vr_config / 模型加载逻辑，并启用 OrbitControls，适合确认模型能加载、尺度和朝向是否合理。
- `stereo`：双眼并排预览，用左右眼相机偏移和 scissor viewport 在普通屏幕上近似检查立体渲染、裁剪和深度方向。
- `webxr`：真实 WebXR / SteamVR 路径，启动 SteamVR 后用 PC Chrome / Edge 打开页面并点击 `Enter VR`。

## Payload 协议

推荐通过 URL payload 指定 BrainDance 模型：

```text
https://127.0.0.1:5174/?payload=<encoded-json>
```

payload 示例：

```json
{
  "ply": "https://example.com/point_cloud.ksplat",
  "modelUrl": "https://example.com/point_cloud.ksplat",
  "poses": "https://example.com/webgl_poses.json",
  "posesUrl": "https://example.com/webgl_poses.json",
  "matrix": [1, 0, 0, 0],
  "imageId": "frame_000123.jpg",
  "sceneId": "my-room",
  "previewMode": "webxr",
  "authSession": {
    "userId": "u_001",
    "email": "demo@example.com",
    "displayName": "Demo User"
  },
  "modelList": [
    {
      "id": "room-a",
      "name": "Room A",
      "modelUrl": "https://example.com/room-a.ksplat",
      "posesUrl": "https://example.com/webgl_poses.json",
      "tags": ["room", "scan"]
    }
  ],
  "markers": [
    {
      "id": "desk",
      "label": "Desk",
      "position": [0, 1.2, -2.4]
    }
  ],
  "searchResults": [
    {
      "id": "hit_001",
      "label": "Desk close-up",
      "markerId": "desk",
      "score": 0.91
    }
  ]
}
```

本端不是把 VR 渲染器接到 Flutter，而是在网页 VR 内复刻 Flutter 客户端的核心用户态和交互入口。保留这些全局 hook 是为了让桌面调试、WebView 或后端 Recall 结果能用同一套协议灌入网页：

```ts
window.loadModelFromFlutter({
  ply: 'https://example.com/point_cloud.ksplat',
  poses: 'https://example.com/webgl_poses.json',
  authSession: { displayName: 'Demo User' },
  modelList: [{ id: 'a', name: 'A', modelUrl: 'https://example.com/a.ksplat' }],
})

window.setBrainDanceSession({ displayName: 'Demo User' })
window.setModelListForTimePeeling([{ id: 'a', name: 'A', modelUrl: 'https://example.com/a.ksplat' }], 'a')
window.setRecallQuery('桌子')
window.setRecallSearchResults([{ id: 'hit', label: '桌子', markerId: 'desk' }])
window.setRecallMarkers([{ id: 'desk', label: '桌子', position: [0, 1.2, -2.4] }])
```

字段兼容 Flutter 侧历史命名：`modelUrl / ply`、`posesUrl / poses`、`timePeelingModels / modelList`、`authSession / session / userSession`、`recallMarkers / markers`、`recallSearchResults / searchResults`。

## VR 配置

Viewer 会根据 `poses` URL 推导同目录的 `vr_config.json`。如果加载失败，则使用 `public/models/vr_config.json` 中的默认值。

```json
{
  "worldScale": 1.0,
  "worldPosition": [0, 0, -2.2],
  "worldRotationY": 0,
  "userHeight": 1.6,
  "startDistance": 2.2,
  "near": 0.01,
  "far": 2000,
  "preferCompressedModel": true
}
```

网页壳层包含用户状态、模型列表、搜索结果、空间 marker、加载进度和 VR HUD。键盘快捷键：

- `1`：切换 desktop preview
- `2`：切换 stereo preview
- `3`：切换 webxr mode
- `[`：缩小模型
- `]`：放大模型
- `Q / E`：旋转模型
- `WASD`：移动调试相机
- `R`：重置位置和缩放

WebXR 控制器映射：

- 左摇杆：水平漫游
- 右摇杆：转向和升降
- `A / X` 或等价侧键：重置当前场景
- `B / Y` 或等价侧键：显示 / 隐藏 VR HUD
- 单手 `Grip`：抓取移动和旋转场景
- 双手 `Grip`：按双手距离缩放场景

`desktop` 和 `stereo` 模式用于开发调试，不能替代 SteamVR + PICO Neo 2 的真实头显验证。

## 模型格式

VR 优先加载压缩格式。传入 `point_cloud.ply` 时会按顺序尝试：

1. `point_cloud.ksplat`
2. `point_cloud.splat`
3. 原始 `point_cloud.ply`

大型 `.ply` 在 VR 双眼渲染下可能加载慢或帧率低，真实演示建议优先准备 `.ksplat`。

## 坐标系修正

VR viewer 的 3DGS 加载链路以 `3dgs_viewer/my-3dgs-viewer` 为主参考，而不是 Spark viewer。模型加载时在 `addSplatScene` 的 `scale` 上统一应用 `[worldScale, worldScale, -worldScale]`，只在加载层做 Z 轴镜像，保持 XY 水平面不再额外翻转。

Recall marker / 搜索结果传入的矩阵和位置也会进入同一套 Z 轴转换。相机跳转时会先套用当前 `splatMesh.matrixWorld` 再分解位姿，避免模型镜像后出现跳转到负 Z 或上下反向的问题。
