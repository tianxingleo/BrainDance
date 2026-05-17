# spark-3dgs-viewer

`spark-3dgs-viewer/` 是 BrainDance 基于 `@sparkjsdev/spark` 的独立 Web 查看器目录，与现有 `my-3dgs-viewer/` 并列，互不覆盖。

## 当前定位

- 不替换原有查看器
- 保持与 Flutter 现有 `payload` 协议兼容
- 验证 `Spark` 是否能承接模型加载、位姿跳转、局部特效与打印纸板 AR 挂载

## 当前已实现

- `Spark` 渲染内核接入
- `window.loadModelFromFlutter(...)` 兼容
- URL `payload` / `ply` / `poses` / `matrix` / `imageId` 启动兼容
- 位姿列表、标签检索、镜头跳转
- 焦距控制
- 一个基于 `SplatEdit + SplatEditSdf` 的球形局部高亮特效
- `mode=marker-ar` 的打印纸板 AR 模式
- MindAR 图片识别锚点 + `SplatMesh` 挂载
- AR 缩放、Y 轴旋转、高度偏移与重置控制

## Marker AR 模式

### URL 参数

```bash
/?mode=marker-ar \
  &model=https://example.com/point_cloud.ksplat \
  &target=https://example.com/braindance-card.mind \
  &scale=0.25 \
  &rx=-1.5708 \
  &ry=0 \
  &rz=0 \
  &ox=0 \
  &oy=0.04 \
  &oz=0
```

### 默认资源

- 默认模型：`/models/scene_auto_sync_raw.ksplat`
- 默认目标图描述文件：`/targets/braindance-card.mind`
- 目标图资源说明：`public/targets/README.md`

### 当前交互

- 识别到打印纸板后，将 3DGS 模型挂到 marker anchor 下
- 底部控制条支持缩放、左右旋转、上下偏移、重置
- 顶部与底部状态条提示权限、识别、丢失追踪等状态

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

- 当前仓库未附带真实 `.mind` 二进制描述文件，接入实际打印卡片前需要自行生成并放入 `public/targets/`
- 第一版优先面向移动浏览器外部打开，不保证所有嵌入式 WebView 摄像头权限链路已打通
- 移动端建议优先使用 `.ksplat`，避免直接加载超大 `.ply`
- 当前 AR 变换参数仅保存在前端运行态，后续可继续接入 `model_assets.ar_meta` 做持久化
