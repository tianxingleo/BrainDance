# spark-3dgs-viewer

`spark-3dgs-viewer/` 是 BrainDance 为 `@sparkjsdev/spark` 单独开出的备选 Web 查看器目录，与现有的 `my-3dgs-viewer/` 并列，互不覆盖。

## 当前定位

- 不替换原有查看器
- 保持与 Flutter 现有 `payload` 协议兼容
- 验证 `Spark` 是否能承接模型加载、位姿跳转和局部特效

## 当前已实现

- `Spark` 渲染内核接入
- `window.loadModelFromFlutter(...)` 兼容
- URL `payload` / `ply` / `poses` / `matrix` / `imageId` 启动兼容
- 位姿列表、标签检索、镜头跳转
- 焦距控制
- 一个基于 `SplatEdit + SplatEditSdf` 的球形局部高亮特效

## 本地运行

```bash
cd 3dgs_viewer/spark-3dgs-viewer
npm install
npm run dev
```

生产构建：

```bash
npm run build
```

## 注意

- 这是实验性备选查看器，不影响 `my-3dgs-viewer/`
- 当前位姿矩阵与 `Spark` 的相机坐标系是否完全一致，还需要结合真实模型继续验证
- 默认保留了与原查看器相近的 UI 壳层，但渲染内核和交互实现已经切到 `Spark`
