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

## 说明

这个目录只负责 Web 端查看器本身。完整的模型生成、位姿导出和同步脚本位于上级目录 [3dgs_viewer/README.md](/home/ltx/projects/BrainDance/3dgs_viewer/README.md)。
