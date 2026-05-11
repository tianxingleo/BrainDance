# my-3dgs-viewer

`my-3dgs-viewer/` 是 BrainDance 当前使用的 Web 端 3DGS 查看器，基于 `Vue 3 + Vite + Three.js`，核心组件是 `src/components/GaussianViewer.vue`。

## 当前能力

结合现有代码，查看器已经包含：

- 3D Gaussian Splatting 场景加载
- Arcball 交互控制
- 镜头位姿读取与飞跃
- 基于标签的简单搜索与跳转
- 自动旋转
- 调试信息与手动相机微调
- 焦距面板
- WebXR / VR 入口检测
- VR 手柄漫游 / 转向 / 视角重置
- VR 内置 HUD 状态面板
- 统一的场景轴系修正：加载层对模型做 Z 轴镜像，避免模型落到错误半轴

入口文件：

- `src/App.vue`
- `src/components/GaussianViewer.vue`

## 依赖

项目当前关键依赖见 `package.json`：

- `vue`
- `three`
- `@mkkellogg/gaussian-splats-3d`
- `gsap`
- `@sparkjsdev/spark`

## 本地运行

```bash
cd 3dgs_viewer/my-3dgs-viewer
npm install
npm run dev
```

生产构建：

```bash
npm run build
```

预览构建结果：

```bash
npm run preview
```

## 模型文件位置

查看器默认从 `public/models/` 读取相关资源，当前仓库里已经包含示例文件：

- `public/models/webgl_poses.json`
- `public/models/webgl_poses_with_tags.json`
- `public/models/transforms.json`

如果要接入新的模型资源，通常需要同步以下内容：

- 高斯模型文件
- 位姿 JSON
- 可选的参考图片与标签数据

如果某个模型是单图 SHARP 这类“只有模型、没有额外镜头信息”的产物，加载时可以只传 `.ply`/`.ksplat`/`.splat`，不要再复用上一场景的 `poses`。查看器会在这类切换里自动清空旧位姿，避免把旧镜头逻辑套到当前模型上。

当前查看器的 loading 遮罩只覆盖“模型解析和首帧可见”阶段；模型一旦可渲染，遮罩就会先关闭，后续的位姿读取或入场动画不会继续挡住画面。

如果 Flutter 侧没有传入 `poses`，viewer 会按“单模型资源”处理，不再自动补一个 `webgl_poses.json`。

如果传了 `poses` 但解析不出有效的镜头数组，viewer 也会直接把它当成无位姿场景，不会再把原始 JSON 继续喂给相机逻辑。

## 3DGS 渲染性能策略

查看器会在传入 `.ply` 时优先探测同名 `.ksplat` 和 `.splat`，存在则优先加载优化格式，失败后自动回退原始 URL。生产流水线建议同时保留原始 `.ply` 归档，并额外产出 Web 优先的 `.ksplat`。

排序与内存传输使用自适应配置：

- 页面满足跨域隔离时启用 `sharedMemoryForWorkers` 与 `gpuAcceleratedSort`。
- Vite 本地开发已配置 `Cross-Origin-Opener-Policy` 和 `Cross-Origin-Embedder-Policy`，线上 CDN / WebView 容器也需要提供等效响应头。
- 静态模型默认启用 `optimizeSplatData`、`freeIntermediateSplatData`、`halfPrecisionCovariancesOnGPU` 等渲染路径优化。
- 入场粒子动画仍保留，但会按桌面和移动设备预算采样，避免大模型首次加载时额外生成过大的 `Points` 几何体。

## 说明

这个目录只负责 Web 端查看器本身。完整的模型生成、位姿导出和同步脚本位于上级目录 [3dgs_viewer/README.md](/home/ltx/projects/BrainDance/3dgs_viewer/README.md)。

## VR 备注

- 进入 VR 后，左摇杆负责前后与平移，右摇杆负责转向与垂直位移。
- `A/X` 可重置视角，`B/Y` 可退出 VR。
- 当前模型加载层已经统一施加 Z 轴镜像修正，用来对齐导出模型与查看器坐标系；如果后续更换导出协议，应优先改这一处，而不是在多个镜头函数里分散补丁。
