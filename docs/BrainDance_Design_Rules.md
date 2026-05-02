# BrainDance 设计规范与交互原则 (Anti-AI-Templates)

这套规范的作用是为了让 BrainDance 从“标准的 AI 科技模板”中抽离，变成一个真实的**空间记录工具**（Spatial Archive / Scanner Workbench）。拒绝泛泛的炫技设计，追求克制、严肃与物理仪器反馈感。

## 🚫 绝对禁止项 (Do NOT Do)
1. **禁止使用大面积荧光紫、电蓝渐变**（抛弃 AI 会用的常见主题）。
2. **禁止使用大面积的玻璃拟态（Glassmorphism）和超大粗白泛光阴影**，拒绝漂浮的炫技界面。
3. **禁止采用居中对称的大 Hero 标题**（诸如“重塑你的未来空间”一类的空话）。
4. **禁止所有的模块和卡片用一样的“过度平滑圆角+轻微边框”**。它需要硬核、有锋利感的工作台调性。
5. **禁止单纯为了“炫酷”在背景里放置漂浮的 Blob / 呼吸粒子**。

## ✅ 必须遵守的风格 (Must Do)
1. **真实仪器质感**：界面需要具备档案室与仪器的复合质感。字体多使用 Courier 表现时空序列或设备 ID。
2. **不对称的工作区布局**：例如使用大量左对齐、硬边或只圆一侧的包裹框。这增强了工具感。
3. **针对动作的动画**：动画仅作为物理反馈，而不是展示用。例如快门按下时的物理收缩、抖动时的快速心跳预警频闪、进度条的刻度推进。

## 🎨 Token 与色彩系统 (Design Tokens)
全局使用 BDDesign 统一调用：
- **颜色系统**：
  - paperWhite (纸白 #F9F9F8) - 普通状态下的背景基底或重要主色。
  - shGray (石灰 #EDEDEA) - 分割线、次级背景。
  - inkBlack (墨黑 #1E1E20) - 工具面板、文字或强调块背景。
  - mutedBlue (钝蓝灰 #6B7A8F) - 系统提示、状态亮色。
  - darkRed (暗红棕 #8B4747) - 越界、预警、错误。
  - adedOlive (褪色橄榄绿 #6D8260) - 成功、进度完成。

- **动作曲线**（BDMotion）：
  - 物理段落动作不拖泥带水，采用迅速的 durationFast。

## 玻璃表面组件规范（2026-05-02）

本项目可以借鉴 Kyant AndroidLiquidGlass / Backdrop 的分层思路，但 Flutter 端不直接依赖 Android Compose 或 Wear Material3。当前统一通过 `app/lib/widgets/bd_surfaces.dart` 中的 `BDGlassSurface` 和 `BDPanelCard(glass: true)` 管理毛玻璃表面。

### 使用原则

- 优先把玻璃用于悬浮导航、搜索入口、底部弹层、设置页主面板、Recall 关键结果面板等少量高层级容器。
- 不要在长列表的每一个 item 上默认启用真 `BackdropFilter`，避免滚动时产生过多离屏渲染开销。
- 普通内容卡片继续使用 `BDPanelCard` 默认实体面板；只有需要透出背景层次时才传入 `glass: true`。
- 新增玻璃效果时不要手写 `ClipRRect + BackdropFilter + BoxDecoration`，应先复用 `BDGlassSurface`，确需差异化再通过参数覆盖 `blurSigma`、`tintColor`、`borderColor` 或 `shadows`。

### 分层结构

- 背景采样层：由 `BackdropFilter` 对背后内容做局部模糊，`panel` 默认较轻，`floating` 默认更强。
- 可读性染色层：浅色模式使用纸白半透明，暗色模式使用深色 surface 半透明，保证文字和图标对比度。
- 边缘高光层：使用低透明度边框和左上到右下的轻微高光渐变，避免廉价大面积玻璃拟态。
- 投影层：只保留低透明度阴影，强调悬浮关系，不制造厚重发光感。

### 已接入位置

- `app/lib/floating_nav_bar.dart`：底部悬浮导航改为 `BDGlassSurface(variant: BDGlassVariant.floating)`。
- `app/lib/pages/recall/search_header_section.dart`：Recall 搜索框、Agent 提示面板和搜索模式弹层接入统一玻璃表面。
- `app/lib/pages/recall/overview_card.dart`、`app/lib/pages/recall/processing_section.dart`：Recall 关键概览与处理面板启用 `BDPanelCard(glass: true)`。
- `app/lib/pages/settabs/`：设置页主要面板启用统一玻璃卡片，保持与悬浮导航一致的材质语言。
