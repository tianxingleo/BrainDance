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
https://127.0.0.1:5174/
```

建议先启动 SteamVR，再用 PC Chrome / Edge 打开页面并点击 `Enter VR`。WebXR 需要安全上下文，本端通过 Vite basic SSL 在开发环境提供 HTTPS。

## Payload 协议

推荐通过 URL payload 指定 BrainDance 模型：

```text
https://127.0.0.1:5174/?payload=<encoded-json>
```

payload 示例：

```json
{
  "ply": "https://example.com/point_cloud.ksplat",
  "poses": "https://example.com/webgl_poses.json",
  "matrix": [1, 0, 0, 0],
  "imageId": "frame_000123.jpg",
  "sceneId": "my-room"
}
```

同时保留 Flutter WebView 兼容入口：

```ts
window.loadModelFromFlutter({
  ply: 'https://example.com/point_cloud.ksplat',
  poses: 'https://example.com/webgl_poses.json',
})
```

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

第一版支持桌面面板的缩放、重置和状态显示。键盘快捷键：

- `[`：缩小模型
- `]`：放大模型
- `R`：重置位置和缩放

## 模型格式

VR 优先加载压缩格式。传入 `point_cloud.ply` 时会按顺序尝试：

1. `point_cloud.ksplat`
2. `point_cloud.splat`
3. 原始 `point_cloud.ply`

大型 `.ply` 在 VR 双眼渲染下可能加载慢或帧率低，真实演示建议优先准备 `.ksplat`。
