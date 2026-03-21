# Spark 融入 3DGS 查看器的集成评估

## 核对结果

已核对以下信息源：

- GitHub 仓库：`https://github.com/sparkjsdev/spark`
- 官方文档：`https://sparkjs.dev/`
- 本地已安装包：`3dgs_viewer/my-3dgs-viewer/node_modules/@sparkjsdev/spark`

这次不是只看 README 标题，而是进一步确认了本地包暴露的真实 API，核心点包括：

- `SplatMesh`
- `SparkControls`
- `SplatEdit`
- `SplatEditSdf`
- `SplatEditSdfType`

这说明 `Spark` 的确提供了可运行时编辑 splat 的正式能力，不是单纯“能加载一个 splat 文件”的轻量查看器。

## 结论

可以融入，而且很适合用来做 3DGS 的动态特效。

但要注意一件事：

- `@sparkjsdev/spark` 不是当前 `@mkkellogg/gaussian-splats-3d` 的一个“特效插件”
- 它本身就是另一套 `Three.js` 里的 3D Gaussian Splatting 渲染与编辑管线

所以真正可行的接入方式是：

- 用 `Spark` 替换当前查看器的 splat 渲染内核
- 保留现有的 UI、Flutter 通信协议、位姿跳转逻辑
- 先做一个实验分支或开关模式，再决定是否全面切换

不建议的方式：

- 在当前 `GaussianSplats3D.Viewer` 之上直接叠加 `Spark` 做特效

这样会出现双渲染器、双相机状态、双排序逻辑，维护成本高，而且很多高级能力根本用不上。

## 本项目当前现状

当前查看器核心在：

- `3dgs_viewer/my-3dgs-viewer/src/components/GaussianViewer.vue`

当前渲染内核是：

- `@mkkellogg/gaussian-splats-3d`

当前代码里已经安装了：

- `@sparkjsdev/spark`

但目前没有真正使用。

## 为什么 Spark 值得接

`Spark` 适合这个项目，不是因为它“能显示 splat”，而是因为它能对 splat 做运行时编辑。

它能直接支持的方向包括：

- 实时改 splat 颜色
- 实时改透明度
- 让 splat 在局部区域发生位移
- 基于 SDF 形状做局部高亮、隐藏、挤压、扩散
- 做 GPU 上的 splat 生成和修改
- 做多视点和和普通 mesh 混合渲染

这些能力正好适合 BrainDance 这种“镜头跳转 + 语义检索 + 空间交互”的查看器。

从本地类型定义和官方文档看，`Spark` 当前至少已经明确支持：

- `new SplatMesh({ url })` 直接加载 `.ply/.spz/.splat/.ksplat/.sog`
- `SparkControls` 负责相机交互
- `SplatEdit + SplatEditSdf` 做局部编辑
- `SplatMesh.onFrame(...)` 做逐帧动画
- splat 和普通 `Three.js` mesh 混合渲染

这几个点正是本项目最关心的“查看 + 交互 + 特效”基础设施。

## 可以做出来的特效

如果切到 `Spark`，比较实用的特效包括：

- 语义检索命中区域高亮
- 从检索点位向外扩散的聚光效果
- 选中镜头附近的 splat 染色
- 参考图对应区域的局部强调
- 镜头飞跃时的路径拖尾或空间波纹
- 按标签给不同区域做颜色编码
- 用 SDF 做“切开查看 / 剖切 / 局部隐藏”
- 用位移做呼吸、脉冲、热力扰动、风场形变

## 为什么不能低成本直接“塞进去”

当前 `GaussianViewer.vue` 和旧渲染器耦合较深，主要体现在：

- 用 `new GaussianSplats3D.Viewer(config)` 初始化整套 viewer
- 用 `viewer.addSplatScene(...)` 加载模型
- 相机、controls、渲染循环都通过 `viewer` 管
- 代码里多次直接访问 `viewer.camera`、`viewer.controls`、`viewer.renderer`
- 特效 shader 注入依赖旧渲染器内部 material
- `viewer.getSplatMesh()` 被拿来生成粒子动画和做 shader hack

这意味着如果要用 `Spark` 的高级编辑能力，迁移点不是单一函数，而是整条渲染链。

换句话说：

- 现在的 `GaussianSplats3D.Viewer` 是“黑盒 viewer”
- `Spark` 更像“你自己组 scene / camera / renderer / controls / splat mesh 的底层渲染框架”

所以它更灵活，但也意味着你得自己接回当前查看器的大部分控制逻辑。

## 迁移后哪些东西可以保留

可以直接保留或稍作改造后保留的部分：

- Flutter 和 Web 页面之间的 `payload` 协议
- `loadModelFromFlutter(...)`
- 位姿 JSON 加载逻辑
- 标签搜索与镜头列表 UI
- `flyToImage(...)` 的位姿匹配思路
- 焦距面板和 FOV 计算
- 外层界面样式

需要替换的部分：

- splat 模型加载
- 渲染循环
- 相机控制器
- 旧版 shader 注入逻辑
- 基于 `viewer.getSplatMesh()` 的内部调用

## 最稳妥的接入路线

### 第一阶段：做 Spark 实验版查看器

目标：

- 新建一个 `SparkGaussianViewer.vue`
- 只完成基础加载、相机、位姿跳转、标签 UI 复用
- 先证明 `.ply` 资产和当前位姿数据可用

验收标准：

- 现有模型能正常显示
- 能根据位姿 JSON 飞到指定视角
- Flutter 仍能通过原有协议打开查看器

### 第二阶段：补齐交互能力

目标：

- 对齐当前自由模式 / Orbit 模式
- 恢复焦距控制
- 恢复 FPS、加载态、错误态

### 第三阶段：加真正的 Spark 特效

建议先做三类最有价值的：

- 检索高亮
- 局部染色
- SDF 剖切 / 隐藏

这三类最容易体现价值，也最适合和 BrainDance 的语义检索结合。

## 推荐优先做的两个特效

### 1. 检索命中区域高亮

做法：

- 根据标签、位姿或语义定位到一个空间中心点
- 用 `SplatEdit + SplatEditSdf` 在该区域做颜色增强和透明度控制
- 优先使用 `SPHERE` 或 `ELLIPSOID` 类型做第一版，最容易和“命中点 + 半径”模型结合

收益：

- 用户检索“桌面左侧”“门口”“椅子旁边”时，查看器能直接给出空间提示

### 2. 剖切查看

做法：

- 用 `PLANE` / `BOX` 类型的 SDF 编辑 splat
- 对区域外 splat 降透明度或直接隐藏

收益：

- 看室内、设备、结构层次时非常直观

## 风险

主要风险不是“能不能跑”，而是迁移工作量。

风险点：

- 当前查看器围绕旧 renderer 封装较深
- `Spark` 的相机与渲染接法和旧 viewer 不同
- 现有自定义 shader 动画需要改写成 `Spark` 的 modifier / edit 方式
- 需要重新验证移动端和 Flutter WebView / 外部浏览器模式表现

另外还有两个现实问题需要提前验证：

- 当前模型是否全部能被 `Spark` 稳定读取，尤其是你们实际产出的 `.ply` 变体
- 当前位姿矩阵和坐标系是否能和 `Spark + Three.js` 相机姿态一一对齐，避免再次出现朝向翻转问题

## 建议

建议继续推进，但方式要对：

- 不要在现有 `GaussianSplats3D.Viewer` 上硬叠 `Spark`
- 直接做一个 `Spark` 版实验查看器
- 先以“基础查看 + 一个高亮特效”作为最小可行版本

如果实验版跑通，再逐步替换现在的默认查看器。

## 最小实验版建议

如果马上开始做，建议最小范围不要超过下面四项：

- 新建 `SparkGaussianViewer.vue`
- 用 `SplatMesh` 替代现有 `GaussianSplats3D.Viewer`
- 复用当前 `payload` 协议和 `flyToImage(...)` 逻辑
- 只先实现一个 `SplatEditSdf` 球形高亮特效

这样最容易在一两轮内判断三件事：

- 现有资产能不能直接跑
- 位姿跳转能不能对齐
- `Spark` 特效是不是足够值回迁移成本

## 下一步建议

下一步最合理的是直接开始做一个最小实验版，范围控制在：

- 读取当前 `ply`
- 读取当前 `poses`
- 实现 `flyToImage`
- 用 `Spark` 加一个局部高亮特效

这个版本一旦跑通，就能快速判断是否值得全面迁移。
