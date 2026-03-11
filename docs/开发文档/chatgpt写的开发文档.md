下面这版我已经按“**当前分支代码实装 > `tianxingleo-DLUT-L20` 分支 README > `docs/API_DOC.md` > 你上传的项目计划书**”替你校过口径了。先提醒你两个很关键的冲突：根 README 还写着 `app/` 尚未纳入仓库，但当前分支实际上已经有完整 Flutter 工程；同时 `app/README.md` 仍然还是默认的 “A new Flutter project.”。所以开发文档里凡是“是否已实现”的判断，建议一律以当前分支代码为准。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

项目计划书更适合给你提供“项目背景、应用场景、价值、人群”的写法，不适合直接拿来证明“这个功能现在已经做完了”。也就是说，**项目价值看计划书，功能完成度看当前分支代码**。

先统一一下你文档里的标记方式，提交前再删掉这些标签：

| 标记             | 含义                                            |
| ---------------- | ----------------------------------------------- |
| `[已实现]`       | 当前分支能看到明确代码路径                      |
| `[已接入待联调]` | 后端/接口已存在，但移动端或闭环还没真正打通     |
| `[方案预留]`     | README/计划书里提到，但当前分支里不宜写成已落地 |
| `<u>待确认</u>`  | 技术口径容易被老师追问，建议保守写法            |

【本轮新增 2026-03-09】本次补充采用以下额外标记：

| 标记                      | 含义                                                       |
| ------------------------- | ---------------------------------------------------------- |
| `【本轮新增 2026-03-09】` | 本次新增内容（补全案例、补全优化细节）                     |
| `【本轮修订 2026-03-09】` | 对原有表述的纠偏（保持原意但口径更贴合代码）               |
| `【待你确认】`            | 代码里暂无可审计实测数据支撑，需你提供最终数值后再定稿     |
| `【代码依据】`            | 当前仓库真实实现位置（文件:行号），便于答辩时快速展示证据  |

### 先给你一个“当前分支真实状态表”

下表是按当前分支代码、README 和 API_DOC 对出来的，适合直接转成你文档里的“当前完成情况/模块成熟度说明”。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

| 模块                    | 当前状态         | 建议在开发文档中的表述                                    |
| ----------------------- | ---------------- | --------------------------------------------------------- |
| 登录与应用入口          | `[已实现]`       | 已完成 Supabase 初始化、登录页与主界面路由切换            |
| 新应用结构（底部四栏）  | `[已实现]`       | 已形成 Recall / Record / Generate / Settings 四主导航结构 |
| “回忆/我的模型”页       | `[已实现]`       | 已支持从 `model_assets` 查询模型列表并展示卡片            |
| 语义搜索                | `[已实现]`       | 已调用 `search-models` Edge Function，支持自然语言查询    |
| 3D 模型查看器           | `[已实现]`       | 已实现基于 WebView 的模型加载、位姿跳转与本地代理         |
| 录制页                  | `[已实现]`       | 已实现相机录制、录制结束跳转提交页                        |
| 录制后“提交任务”        | `[已接入待联调]` | 当前提交页仍只有表单与 Toast，不能写成“已全链路上线”      |
| 生成页（图/文/视频）    | `[已接入待联调]` | 入口 UI 已有，但底部按钮当前仍提示不可用                  |
| `video_3dgs`            | `[已实现]`       | 视频转 3DGS 标准流水线已存在                              |
| `single_image_sam3d`    | `[已接入能力]`   | 单图转 3DGS 后端流水线已接入                              |
| `single_image_sharp`    | `[已接入能力]`   | SHARP 后端流水线已接入，但前端闭环待联调                  |
| `da3_feed_forward_3dgs` | `[已接入能力]`   | DA3 前馈式 3DGS 路径已存在，但对外技术口径宜保守          |
| Time Peeling            | `[方案预留]`     | 适合作为特色/路线图，不建议写成当前闭环功能               |

### 你最该保守写的三句

1. DA3 在当前分支中更适合表述为“替代传统 GLOMAP/COLMAP 位姿—深度解算链路，并在 `da3_feed_forward_3dgs` 路径中支持前馈式 3DGS 导出”；是否进一步概括成“类似 VGGT 的方案”建议由你们项目组最终确认。 API_DOC 把它写成“视频转 3DGS（前馈快速生成）”，而 `DA3Runner` 代码明确写了 “Depth Anything 3 + Streaming”“替代原有的 GLOMAP 解算”，`DA3FeedForwardPipeline` 还写了“绕过 Nerfstudio 训练、直接反投影构建 3DGS”。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))
2. SAM3D 可以放心写成“单图转 3DGS 后端流水线已接入”；SHARP 更稳妥的写法是“后端能力已接入，移动端生成闭环待联调”。 因为 API_DOC 已列出 `single_image_sam3d` / `single_image_sharp` 两类任务，后端也各有独立 pipeline；但前端 `GeneratePage` 当前底部按钮仍直接提示不可用。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))
3. Time Peeling 适合写成“项目特色与下一阶段功能”，不建议在本版开发文档中写成已闭环上线。 README 把它列为核心特性，但当前我核对到的 App 主流程、API_DOC 和 Worker 闭环里，更强的实现证据仍集中在“回忆列表—搜索—查看器—异步重建”这条主链路。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

------

## 可直接贴进模板的开发文档初稿

下面这版是按你给的模板来写的，你可以直接复制到文档里，再把 `<u>待确认</u>` 的地方改掉。

### 1 项目概述

#### 1.1 项目背景

BrainDance（流光·记）是一款面向移动端的“可检索三维记忆应用”。它希望解决传统照片/视频只能记录二维画面、无法保留空间关系与交互视角的问题，把现实空间转化为可浏览、可检索、可回看视角的三维数字资产。当前分支 README 将项目定义为“可检索三维记忆库”，核心方向包括移动端低成本扫描、空间语义检索、时光剥离与端云协同。项目计划书则进一步把它描述为面向“记忆留存、空间消失记录、数字海马体”的产品。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

#### 1.2 项目定位

##### 1.2.1 应用场景（这里可以看看当时写的软创赛的内容，这里不详细）

本项目主要面向个人空间留存、老房间/宿舍/街区的三维记录、可语义检索的空间档案管理，以及未来 XR 终端的原生空间内容生产。对于用户而言，它不只是“拍一段视频”，而是把一个场景保存成可再次进入、可再次搜索的三维记忆。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

##### 1.2.2 目标人群（这里可以看看当时写的软创赛的内容，这里不详细）

目标人群包括希望低成本记录生活空间的普通用户、需要对空间变化进行留存的文化/城市记录者，以及希望获得可检索三维场景资产的空间内容创作者。若从项目价值角度扩展，还可面向有怀旧回溯需求的家庭或照护场景，但这一部分更适合作为应用延展，而不是当前分支的已交付功能。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

#### 1.3 项目方案

项目采用“Flutter 移动端 + Supabase BaaS + Python Worker”的端云协同架构。移动端负责登录、模型浏览、录制、生成入口与查看；Supabase 提供鉴权、对象存储、任务表、模型表、Realtime 与 Edge Function；Python Worker 负责异步消费任务，执行 `video_3dgs`、`single_image_sam3d`、`single_image_sharp`、`da3_feed_forward_3dgs` 等流水线，并将模型、预览图和元数据写回存储与数据库。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

#### 1.4 项目目标（这里好像只说了现阶段的目标没说后阶段的目标，具体参考软创赛文档的核心创新点以及软创赛初赛文档文末未来可拓展部分）

本项目的阶段性目标包括：
（1）形成一个可运行的移动端三维记忆应用结构；
（2）完成视频/图片到三维模型的异步任务闭环；
（3）支持用户在“我的模型”中浏览资产，并通过自然语言完成语义搜索；
（4）实现基于移动端 WebView 的模型查看与定位视角跳转；
（5）为 Time Peeling、单图快速生成、XR 适配等后续能力预留架构接口。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

#### 1.5 项目价值

项目的核心价值是把“二维相册”升级为“空间记忆资产库”，降低 3D 内容生产门槛，并为未来的空间计算内容生态提供原生数据来源。从产品视角看，它把“记忆保存”从平面内容提升为可交互空间；从工程视角看，它把移动端采集、BaaS 调度、AI 重建与语义检索串成了一个统一流程。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

------

### 2 开发计划

#### 2.1 最终呈现形式

作品最终呈现为一套移动端 App 与配套云端处理系统。移动端采用四主导航结构：Recall（回忆）、Record（录制）、Generate（生成）、Settings（设置）；二级页面包括 Login、VideoSubmit、WebGLViewer 等。当前分支已明确体现这一“新应用结构”。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

#### 2.2 主要功能描述

可将主要功能写成如下表格。下表依据当前分支代码、README 和 API_DOC 整理。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

| 功能模块   | 功能           | 功能描述                                       | 优先级 |
| ---------- | -------------- | ---------------------------------------------- | ------ |
| 用户与鉴权 | 登录/会话管理  | 初始化 Supabase，按 session 进入登录页或主界面 | 高     |
| 回忆库     | 模型列表       | 直接查询 `model_assets`，按时间展示三维资产    | 高     |
| 空间检索   | 语义搜索       | 调用 `search-models`，返回模型与相似结果       | 高     |
| 3D 查看    | 模型渲染       | 使用 WebView + 本地代理 + JSBridge 加载模型    | 高     |
| 采集模块   | 视频录制       | 调用相机录制视频，跳转提交页                   | 高     |
| 任务调度   | 异步重建       | 写入 `processing_tasks`，由 Worker 异步处理    | 高     |
| 单图生成   | SAM3D / SHARP  | 后端流水线已接入，前端闭环待联调               | 中     |
| 时间维度   | Time Peeling   | 同坐标系多时刻叠加，当前建议写为规划项         | 中     |
| 系统设置   | 主题/语言/账户 | 管理 App 配置与用户状态                        | 中     |

#### 2.3 运行环境

运行环境建议写成“软件环境 + 硬件环境”两张表。相关信息可直接取当前分支 README 和 `pubspec.yaml`。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

| 类别 | 项目                                                         | 简要说明                              |
| ---- | ------------------------------------------------------------ | ------------------------------------- |
| 软件 | Flutter SDK 3.10+（当前 app sdk 为 `^3.10.7`）               | 移动端开发框架                        |
| 软件 | `supabase_flutter`                                           | 负责鉴权、数据库、存储、函数调用      |
| 软件 | `camera` / `image_picker` / `photo_manager`                  | 负责录制与选取素材                    |
| 软件 | `webview_flutter`                                            | 负责 3D 查看器承载                    |
| 软件 | `flutter_riverpod`                                           | 状态管理                              |
| 软件 | Python 3.10+                                                 | Worker 与 AI 流水线                   |
| 软件 | Docker Desktop / Supabase CLI                                | 本地基础设施                          |
| 硬件 | OPPO Find X8                                                 | 当前 README 中给出的移动端测试设备    |
| 硬件 | Intel i5-12600KF / 64GB / RTX 5070 12GB（**这个不正确了，我现在更新了设备：  当前 AI Engine 测试/推荐服务器（本机，2026-03-09 实测） CPU: Intel Xeon Platinum 8260 × 2（双路，96 线程） 内存: 503GiB（约 512GB） 显卡: NVIDIA L20 46GB × 2（双卡） 操作系统: Ubuntu 22.04.5 LTS（Kernel 6.8.0-100-generic））** | 当前 README 中给出的服务器/开发机配置 |
| 硬件 | NVIDIA GPU**~~（CUDA 11.8+）~~**   **实际上是cuda12.8**      | AI Worker 重建环境要求                |

【本轮修订 2026-03-09】建议在正式提交版把硬件段直接改成如下三行，避免评委看见“旧配置+备注”的痕迹：

- AI Engine 训练/推理服务器（2026-03-09 实测）：`Intel Xeon Platinum 8260 ×2`、`503GiB RAM`、`NVIDIA L20 46GB ×2`、`Ubuntu 22.04.5 LTS (Kernel 6.8.0-100-generic)`。
- CUDA 建议口径：`CUDA 12.8`（当前工程环境文档已给出 12.8 口径）。
- 手机端测试机：`OPPO Find X8`、`OPPO Reno 14`（README 已列出）。

【代码依据】`README.md:189-190,195-198`；`ai_engine/3dgs/ENVIRONMENT.md:15,30`

#### 2.4 验收标准

建议写成以下几条：

1. 用户能够正常登录并进入四主导航结构；
2. 用户能够在 Recall 页查看已有模型；
3. 用户能够通过自然语言查询模型并返回结果；
4. 用户能够在移动端查看 3D 模型并执行视角跳转；
5. 后端能够完成异步任务处理，并把结果写回存储与 `model_assets`；
6. 对于失败任务，系统能够保留日志并将状态标记为 `failed`。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

#### 2.5 关键问题（gpt写的关键问题是写文档时候遇到的问题而不是项目遇到的问题，项目遇到的问题。。。有挺多的。我暂时能想到的有1.用户拍摄质量差，我们用ai参与的Pipeline解决，引入opencv、yolo等       2. 3dgs生成速度慢，我们用最新的论文dpeth Anything 3和前馈式3dgs生成来解决，最快实现了2min的多图生成3dgs，用2025年年底的最新论文sharp和sam3d依赖神经网络实现10s单图生成3dgs    3. 这个方向在学术界火热但是实际落地应用很少，很多地方难以寻找参考    4. 高性能专业显卡来之不易      5. 项目复杂度高           6. 3dgs坐标系转换问题，Pipeline中各个环节的坐标系都发生了大量转换，实现前后绑定与最终需要大量的矩阵变换过程，花了很长时间调试  ）

当前最关键的问题有四类：
第一，移动端原始素材质量不稳定，容易影响位姿解算与最终模型质量；第二，大文件上传与模型下发存在耗时与兼容性问题；第三，移动端查看器需要处理 WebView 对 HTTPS/证书与资源访问的限制；第四，README、API_DOC 与代码实装之间存在少量口径差异，文档撰写时必须以代码为准。例如 README 时序图写了 `pop_next_task`，而当前 Worker 代码实际上是轮询 `pending` 任务再更新为 `processing`，所以文档里不建议写死具体 RPC 名称。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

【本轮新增 2026-03-09】可直接补进“关键问题”的细化条目（这部分是你提到的重点）：

1. 拍摄质量波动：移动端视频抽帧后存在模糊、抖动、纹理不足，导致位姿和重建质量不稳定。当前 pipeline 采用 OpenCV 拉普拉斯方差 + 分位数阈值做智能剔除，并在数量过多时做均匀降采样。  
   【代码依据】`ai_engine/3dgs/src/modules/image_proc.py:20,44,51,64`
2. 重建速度与交付格式的平衡：训练链路耗时较长，工程上通过 `delivery_format` 支持 `.ply/.splat/.ksplat`，并在压缩失败时自动回退原始 `.ply`，保证任务可交付而不因压缩环节整体失败。  
   【代码依据】`ai_engine/3dgs/src/core/worker.py:305,315,317,425`
3. 查看器端兼容性：移动端 WebView 对 HTTPS 证书、跨域和本地文件访问有限制，现通过本地 HTTP 服务 + `/proxy/` 代理 + 本地模型缓存规避证书问题并提升二次打开速度。  
   【代码依据】`app/lib/pages/webgl_viewer.dart:74,79,103,198,307,321`
4. 任务链路真实状态与文档口径冲突：`Record -> VideoSubmit` 的前端入口已实现，但 `VideoSubmit` 当前是“表单+Toast”占位，不应写成完整提交闭环。  
   【代码依据】`app/lib/pages/record.dart:111,135`；`app/lib/pages/video_submit.dart:52`
5. 坐标系与矩阵对齐复杂：WebGL 端矩阵需符合 Three.js 列主序，当前实现使用 `c2w.T.flatten()` 输出；该环节出错会直接导致“能加载但视角跳转错位”。  
   【代码依据】`ai_engine/3dgs/src/modules/spatial_anchor.py:210,212`
6. 技术前沿缺少工业化范式：DA3/SAM3D/SHARP 都是新链路，项目中采用“多 pipeline 并行接入 + 工厂分发”的方式降低技术替换成本。  
   【代码依据】`ai_engine/3dgs/src/core/factory.py:9-20`

#### 2.6 进度安排（其中的xr和端侧加速不需要写上去，按照现在的进度感觉有点悬）

这一节不建议完全照搬项目计划书的旧时态。更稳妥的写法是：
“前期已完成基础架构搭建、移动端主导航、模型浏览与 3D 查看器；当前重点推进视频任务提交闭环、单图生成入口联调和搜索能力稳定化；后续计划继续完善 Time Peeling、端侧加速与 XR 适配。”
这样能同时兼顾计划书口径和当前分支真实状态。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

#### 2.7 开发预算（我们的预算主要都是花在了coding plan之类的以及ai 订阅费用，实际上最大的消费是gpu但是学校实验室提供支持，如果要租的话7r/h）

这一节建议写成“示例预算”，并用你们真实条件替换：

- 开发软件：以开源框架为主，预算低；
- 移动端测试设备：以现有设备为主；
- GPU 算力：若使用自有显卡则边际成本较低，若租用云 GPU 则按时计费；
- 对象存储与数据库：早期可用 Supabase 本地环境或免费额度；
- 备份/演示资源：按比赛展示需求补充。
  这里可以加一句：预算以“自有设备 + 开源软件 + 可弹性扩展的云资源”为原则。

------

### 3 可行性分析

#### 3.1 技术可行性分析

技术上，本项目已经形成了较清晰的闭环基础：移动端主结构已存在，Recall 页已经能直接查询 `model_assets` 并调用 `search-models`，WebGLViewer 已完成模型 URL、位姿文件和相机矩阵的桥接，Worker 也已具备任务轮询、流水线执行、模型上传和资产入库能力。因此，作为“移动端三维记忆 + 异步 AI 重建”的系统方案，技术路线是可行的。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/recall.dart))

#### 3.2 资源可行性分析

项目所依赖的主要框架均可获得：Flutter、Supabase、Docker、Python Worker 均有明确环境要求与启动方式；测试设备与服务器配置在 README 中也已列出，说明团队已有一定软硬件基础。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

#### 3.3 市场可行性分析（这个看看计划书）

从项目计划书的定位来看，BrainDance 同时覆盖了个人空间记忆、城市/文化记录、空间计算内容生产等多个方向，具有明确的场景价值。建议在开发文档里把这一节写成“需求存在 + 技术切入点明确 + 当前产品先以核心链路验证为主”，避免写成过重的商业承诺。

------

### 4 需求分析

#### 4.1 数据需求

##### 4.1.1 静态数据

静态数据包括：App 配置项、主题/语言资源、前端本地 WebGL 资源、模型查看器页面资源、演示素材与图标资源。当前 **`pubspec.yaml` 这是啥，我也不知道**。。。。。已配置本地资产目录，移动端查看器也依赖本地 Web 资源启动。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/pubspec.yaml))

【本轮新增 2026-03-09】这里可以加一句非常实用的解释：`pubspec.yaml` 是 Flutter 工程的“资源与依赖清单”，里面声明了字体、图片、WebGL 静态资源等，App 才能在运行时通过 `assets/...` 访问这些文件。这个解释写上去后，老师问“为什么本地 HTML 能被加载”时就有落点。  
【代码依据】`app/pubspec.yaml`；`app/lib/pages/webgl_viewer.dart:127`

##### 4.1.2 动态数据

动态数据包括：用户上传的视频/图片、`processing_tasks` 任务记录、`model_assets` 资产记录、搜索结果、实时日志、模型文件、预览图、`transforms.json` 与 `webgl_poses.json`。这些数据共同构成“采集—重建—检索—展示”的完整链路。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

##### 4.1.3 数据词典

这一节建议直接放表。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

| 数据项             | 类型     | 说明                                                         |
| ------------------ | -------- | ------------------------------------------------------------ |
| `scene_id`         | string   | 场景唯一标识，前端生成                                       |
| `display_name`     | string   | 任务展示名                                                   |
| `task_type`        | string   | `video_3dgs` / `da3_feed_forward_3dgs` / `single_image_sam3d` / `single_image_sharp` |
| `task_params`      | jsonb    | 任务参数，如 `frame_interval`、`conf_threshold`、`delivery_format` |
| `status`           | string   | `pending` / `processing` / `completed` / `failed`            |
| `logs`             | json     | 实时日志数组                                                 |
| `quality_score`    | int      | AI 质量评分                                                  |
| `quality_reason`   | string   | 评分原因                                                     |
| `tags`             | string[] | AI 标签                                                      |
| `description`      | text     | 资产描述                                                     |
| `objects`          | string[] | 关键物体列表                                                 |
| `ply_path`         | text     | 模型在 Storage 中的相对路径                                  |
| `preview_img_path` | text     | 预览图路径                                                   |
| `meta_info`        | jsonb    | 扩展元数据                                                   |

##### 4.1.4 数据采集

视频类任务由移动端相机录制得到，单图或视频生成入口由图片/视频选择器提供素材；原始文件统一上传到 `braindance-assets/{user_id}/{scene_id}/raw/` 下，视频为 `video.mp4`，图片为 `image.png`。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/record.dart))

#### 4.2 功能需求

##### 4.2.1 核心功能模块

功能模块说明建议写成下表。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

| 功能模块 | 功能               | 功能描述                       | 优先级 |
| -------- | ------------------ | ------------------------------ | ------ |
| 用户模块 | 登录与会话         | 管理用户登录状态与访问权限     | 高     |
| 回忆模块 | 模型列表           | 展示历史生成的三维资产         | 高     |
| 搜索模块 | 语义检索           | 输入自然语言查询，返回模型结果 | 高     |
| 查看模块 | 3D 浏览            | 渲染模型并支持视角跳转         | 高     |
| 录制模块 | 视频采集           | 录制原始视频素材               | 高     |
| 提交模块 | 任务创建           | 将素材上传并写入任务表         | 高     |
| 重建模块 | AI 处理            | Worker 消费任务并生成三维资产  | 高     |
| 生成模块 | 单图/文图/视频入口 | 当前以入口与联调为主           | 中     |
| 时间模块 | Time Peeling       | 当前作为规划功能               | 中     |

【本轮新增 2026-03-09】“核心功能模块”建议再补一个“工程小巧思”小表（老师通常爱看这块）：

| 小巧思点 | 具体做法 | 带来的效果 |
| --- | --- | --- |
| 空结果兜底 | Recall 在无数据时自动注入 demo 场景 | 避免首页空白，提高首次可演示性 |
| 搜索异常透传 | 捕获 `FunctionException` 并展示真实错误文本 | 联调时可快速定位后端问题 |
| 提交态保护 | 录制时隐藏底部导航并锁定录制状态 | 减少误触切页导致录制中断 |
| 下载缓存 | WebGL 远程模型下载后缓存到本地并复用 | 二次打开更快、离线回看能力更好 |

【代码依据】`app/lib/pages/recall.dart:297`；`app/lib/pages/record.dart:96,150`；`app/lib/pages/webgl_viewer.dart:177,307`

表2 用例规约，建议至少写两个：

**用例 1：视频生成三维记忆**
（依据当前 API_DOC、Record 页与 Worker 逻辑整理）([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/record.dart))

| 字段         | 内容                                                         |
| ------------ | ------------------------------------------------------------ |
| 用例名称     | 视频生成三维记忆                                             |
| 功能简述     | 用户录制或上传视频，系统异步生成三维模型                     |
| 用例编号     | UC-01                                                        |
| 执行者       | 用户、移动端 App、Python Worker                              |
| 前置条件     | 用户已登录；相机/存储权限正常；网络可用                      |
| 后置条件     | 成功时写入 `model_assets` 并可查看；失败时任务状态为 `failed` 且保留日志 |
| 涉众利益     | 用户希望快速把空间保存为可浏览三维资产                       |
| 基本路径     | 录制视频 → 上传 Storage → 插入 `processing_tasks` → Worker 处理 → 上传模型与预览 → 在 Recall 页查看 |
| 扩展路径     | 上传失败、质量不达标、Worker 执行失败、模型压缩失败后回退原始格式 |
| 字段列表     | `scene_id`、`display_name`、`task_type`、`task_params`、`status` |
| 设计规则     | 原始素材放 `raw/`，输出放 `output/`，前端通过 Realtime 感知状态变化 |
| 未解决的问题 | `<u>移动端录制页到真正任务插入的闭环当前仍待联调</u>`        |
| 备注         | 提交版不要写成“已完全上线”                                   |

**用例 2：语义检索三维记忆**
（依据 Recall 页与 Search API 整理）([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/recall.dart))

| 字段         | 内容                                                         |
| ------------ | ------------------------------------------------------------ |
| 用例名称     | 语义检索三维记忆                                             |
| 功能简述     | 用户输入自然语言，系统返回匹配的三维资产                     |
| 用例编号     | UC-02                                                        |
| 执行者       | 用户、移动端 App、Edge Function                              |
| 前置条件     | 用户已登录；已有模型资产；搜索接口可用                       |
| 后置条件     | 返回结果列表并可进入查看器                                   |
| 涉众利益     | 用户能够“像搜文件一样搜空间”                                 |
| 基本路径     | 输入关键词 → 调用 `search-models` → 返回结果 → 进入 3D 查看器 |
| 扩展路径     | 关键词为空、过长、向量生成失败、数据库查询失败               |
| 字段列表     | `query`、`scene_id`、`description`、`ply_path`、`similarity` |
| 设计规则     | 使用 Authorization；支持自然语言时间过滤                     |
| 未解决的问题 | `<u>搜索结果与位姿跳转的效果仍需持续标定</u>`                |
| 备注         | 可作为本项目特色用例展示                                     |

#### 4.3 性能需求

##### 4.3.1 时间特性

系统应保证前端交互操作不被长耗时重建过程阻塞，任务创建后立即转入异步状态；模型搜索应支持自然语言检索**（这个性能也有，比如我们用了专门的turbo模型）**与时间词解析；模型交付应支持 `.splat`、`.ksplat`、`.ply` 等格式，以便在不同阶段兼顾速度与兼容性。**（比如我们使用压缩算法，将3d模型减小了90%，加载速度从20s变成了2s）**([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))                 **（以及在初赛文档中加入的快慢双链计划，虽然还没有实现但是体现 时效性，时效性是这个比赛的要求）**

【本轮修订 2026-03-09】可在该段后补一个“可审计版本”：

- 搜索链路：意图解析当前使用 `qwen-turbo`，语义向量使用 `text-embedding-v2`，并带时间词解析与日期规范化逻辑。  
  【代码依据】`supabase/functions/search-models/index.ts:342,430,190,284`
- 交付链路：Worker 支持按任务参数选择 `delivery_format`，并在压缩失败时回退，确保任务状态能推进到 `completed/failed` 的明确终态。  
  【代码依据】`ai_engine/3dgs/src/core/worker.py:305,317,425`
- 【待你确认】“模型体积减少 90% / 20s -> 2s”建议在文档里写成“内部实测可显著降低加载时间”，并附你最终实测表（机型、网络、模型大小、冷启动/热启动），避免被问到统计口径时失分。

##### 4.3.2 适应性

移动端运行环境以 Android为主，查看器依赖 `webview_flutter`，当前代码已明确 Flutter Web 和桌面平台不作为主要运行目标，因此比赛文档里可以把“手机端优先适配”写成明确设计策略。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/webgl_viewer.dart))      **（OPPO专门适配）**

【本轮新增 2026-03-09】“OPPO 专门适配”可落成以下两条具体描述：

1. 录制页通过 `cameraRatio/deviceRatio` 做动态缩放，修正不同机型预览拉伸。  
   【代码依据】`app/lib/pages/record.dart:173,176`
2. 查看页在不支持平台直接拦截，明确 Android/iOS 为主战场，减少 Web/桌面环境误用导致的演示事故。  
   【代码依据】`app/lib/pages/webgl_viewer.dart:53-61`

#### 4.4 界面需求

界面风格应突出“回忆浏览、快速录制、生成入口、沉浸式查看”四条主线，主界面采用底部四导航，Recall 页以搜索框和模型卡片为主，Record 页以全屏相机取景为主，Generate 页采用图/文/视频三入口，Viewer 页强调沉浸式查看。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

#### 4.5 接口需求

##### 4.5.1 硬件接口

主要依赖手机摄像头、存储权限、网络连接和 GPU 计算环境；服务端依赖 NVIDIA GPU 与本地/云端存储环境。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

##### 4.5.2 软件接口

软件接口主要包括 Supabase Auth、Database、Storage、Realtime、Edge Function `search-models`，以及 Worker 内部 PipelineFactory、各类 3DGS pipeline、查看器与 JSBridge 之间的接口。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

#### 4.6 其他需求

系统应具备基本的用户隔离能力、对象存储路径规范、失败任务回退与日志保留机制。数据安全层面应依赖 Auth 与 RLS；工程维护层面应使用 `.env` 配置与统一路径规范。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

------

### 5 概要设计

#### 5.1 处理流程

系统主流程可写为：
用户在移动端录制视频或选择图片 → 文件上传到 Supabase Storage 的 `raw/` 目录 → 前端写入 `processing_tasks` → Worker 轮询并锁定任务 → 调用对应 pipeline 生成模型 → 上传 `point_cloud.*`、`transforms.json`、`preview.jpg` → upsert `model_assets` → 用户在 Recall 页浏览、搜索并进入 WebGL 查看器。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

#### 5.2 总体结构设计

总体结构建议画成三层：
第一层是移动端表示层，包含 Login、Recall、Record、Generate、Settings 与 Viewer；
第二层是服务与数据层，由 Supabase 的 Auth、DB、Storage、Realtime 和 Edge Function 构成；
第三层是智能计算层，由 Python Worker 与各类 3DGS/语义分析流水线构成。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

#### 5.3 功能设计

功能设计上，移动端负责“交互与轻量组织”，BaaS 负责“数据与调度”，Worker 负责“重计算与资产入库”。这种划分的好处是：移动端更专注用户体验，Worker 更专注模型生成，而 Supabase 负责中间状态与数据一致性。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

#### 5.4 数据流转设计（realtime之类的吧，可以把我们当时的流程图放过来，不过当时你的流程图是有出现大连理工大学的，这个是规则禁止的，如果没有检查出来会被取消资格的。现在的Pipeline也更新了，应该流程图也可以更新）

这部分你可以重点展开：
原始素材进入 `raw/` 目录；任务元数据进入 `processing_tasks`；Worker 根据 `task_type` 选择视频或图片下载路径；生成结果进入 `output/`；模型索引进入 `model_assets`；语义搜索通过 `search-models` 返回匹配资产；Recall 页基于 `ply_path` 和位姿文件进入查看器。这个部分正好满足你老师要求的“数据流转及展示方式详细说明”。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

【本轮新增 2026-03-09】可以把数据流再展开成“可画图”的 10 步，直接用于流程图重绘：

1. 移动端采集视频/图片；  
2. 前端生成 `scene_id` 与任务参数；  
3. 原始素材上传到 `braindance-assets/{user_id}/{scene_id}/raw/`；  
4. 写入 `processing_tasks`（`pending`）；  
5. Worker 轮询并锁定为 `processing`；  
6. `PipelineFactory` 按 `task_type` 分发 `video_3dgs/sam3d/sharp/da3`；  
7. 生成 `point_cloud.*` + `transforms.json` + `preview.jpg`；  
8. 按 `delivery_format` 压缩与回退上传；  
9. `model_assets` upsert，任务更新为 `completed/failed`；  
10. Recall 查询资产，Viewer 加载模型并按 `webgl_poses` 做视角跳转。  

【代码依据】`ai_engine/3dgs/src/core/worker.py:183,194,259,305,373,425`；`app/lib/pages/recall.dart:683`

#### 5.5 用户界面设计（UI Prototype）

最省事的做法，是直接截当前分支已有页面，再配下面的低保真原型说明。

**原型 1：Recall 首页**

```text
┌──────────────────────────────┐
│  搜索框：输入“红色杯子/上周拍的照片” │
├──────────────────────────────┤
│  模型卡片1   模型卡片2              │
│  模型卡片3   模型卡片4              │
├──────────────────────────────┤
│ Recall | Record | Generate | Settings │
└──────────────────────────────┘
```

**原型 2：Record 录制页**

```text
┌──────────────────────────────┐
│            相机取景区             │
│                                  │
│        [录制按钮] [切换镜头]        │
│        [拍摄提示 / 状态信息]        │
└──────────────────────────────┘
```

**原型 3：WebGL 查看页**

```text
┌──────────────────────────────┐
│        3D 模型渲染区域            │
│                                  │
│   支持模型加载、位姿跳转、全屏查看   │
└──────────────────────────────┘
```

当前分支已有这些页面的代码依据：主界面四导航在 `main.dart`，Recall 页会查询 `model_assets` 并调用 `search-models`，Record 页录制完成后跳转 `VideoSubmitPage`，Viewer 页会把模型 URL 与位姿信息传给 WebView 内部渲染层。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

#### 5.6 数据结构设计（这个让ai阅读sql配合读一下前后端代码一般自己能写好）

建议采用“关系数据 + 对象存储 + 向量检索”的混合结构：

- 关系表：`processing_tasks`、`model_assets`；
- 对象存储：`braindance-assets/{user_id}/{scene_id}/...`；
- 语义检索：基于 `pgvector` 的向量匹配。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

#### 5.7 接口设计

##### 5.7.1 外部接口

外部接口主要包括：
（1）Supabase 登录与鉴权接口；
（2）Supabase Storage 文件上传/下载接口；
（3）`search-models` Edge Function 接口；
（4）移动端对模型公开 URL 的访问接口。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

##### 5.7.2 内部接口

内部接口主要包括：
（1）App 页面与 Supabase SDK 的调用接口；
（2）Worker 与 PipelineFactory 的实例化接口；
（3）Worker 与压缩/上传逻辑的接口；
（4）WebGLViewer 与 WebView JS 的桥接接口 `window.loadModelFromFlutter(...)`。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/ai_engine/3dgs/src/core/worker.py))

#### 5.8 错误/异常处理设计

##### 5.8.1 错误/异常输出信息

系统需要输出上传失败、搜索参数非法、AI 评分不通过、模型压缩失败、查看器不支持平台、任务状态失败等信息。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

##### 5.8.2 错误/异常处理对策

建议写为：

- 上传失败时保留本地素材并提示重试；
- AI 评分不通过时写入 `quality_score` 与 `quality_reason`；
- Worker 失败时将任务标记为 `failed`；
- `.ksplat` 压缩失败时回退到原始 `.ply` 上传；
- 桌面/Web 平台提示用户切换到 Android/iOS。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

【本轮新增 2026-03-09】可再补三条“你们项目里的真实异常处理巧思”：

- 日志同步采用“内存缓冲 + 全量覆盖上传”策略，减少并发覆盖导致的日志丢失。  
  【代码依据】`ai_engine/3dgs/src/core/worker.py:87,100`
- 下载远程模型时若 HTTP 非 200，解析错误体并抛出，前端直接 toast 具体错误，方便现场排障。  
  【代码依据】`app/lib/pages/webgl_viewer.dart:204-211,229`
- 搜索接口异常时优先使用 Edge Function 回传的 `details.error`，避免只显示笼统“请求失败”。  
  【代码依据】`app/lib/pages/recall.dart:297-311`

#### 5.9 系统配置策略

系统配置通过 `.env`、Supabase URL/Key、Worker 存储桶/表名、模型交付格式等统一管理。移动端读取 `.env`，Worker 读取 Supabase 与交付格式相关环境变量。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

#### 5.10 系统部署方案（集群化也可以说一下）

本地部署流程可写为：
`supabase start` 启动基础设施 → `python src/worker.py` 启动 Worker → `flutter run` 启动移动端。
云端部署则可写为：Supabase 承担数据与鉴权，GPU 节点承担 Worker 推理与训练。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

#### 5.11 跨端应用架构设计

当前版本以手机端为核心入口，移动端既负责采集，也负责最终查看；查看器内部通过 WebView 承载 WebGL 页面，因此形成了“原生 Flutter + 内嵌 Web 3D”的跨端混合架构。后续可以扩展到 XR 终端。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/app/lib/main.dart))

#### 5.12 其他相关技术与方案

这一节建议写：

- 直接上传 Supabase Storage，减少中间层；
- 利用 Realtime 感知任务状态；
- 利用 `delivery_format` 支持多种模型交付格式；
- 单图与视频任务共用统一任务表与资产表结构。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

------

### 6 数据库设计

数据库设计可简写为“两张核心表 + 一个核心存储桶”：

1. `processing_tasks`：用于记录任务创建、任务类型、参数、状态、日志、质量评分等；
2. `model_assets`：用于记录生成成功后的模型资产信息；
3. `braindance-assets`：用于保存原始素材、中间文件和最终模型。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

可以补一句：
“系统采用关系表管理任务状态与资产索引，采用对象存储管理大文件，采用向量检索扩展支撑语义搜索。”([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

------

### 7 手机端侧部署设计

#### 7.1 手机环境需求

手机端建议满足以下条件：

- Android 运行环境；
- 可用摄像头与存储权限；
- 可访问 Supabase 网关；
- 具备 `webview_flutter` 所需运行能力。
  当前 README 已给出 OPPO Find X8 与 HUAWEI Mate 30 Pro 作为测试设备；同时查看器代码已明确桌面/Web 不是当前主要部署目标。([GitHub](https://github.com/tianxingleo/BrainDance/tree/tianxingleo-DLUT-L20))

------

### 8 详细设计

#### 8.1 回忆检索与查看模块

##### 8.1.1 功能描述

该模块负责从 `model_assets` 查询历史模型、展示模型卡片，并在用户输入自然语言后调用 `search-models` 完成语义搜索。搜索结果可进一步传递模型路径与位姿信息到 3D 查看器。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/recall.dart))

##### 8.1.2 性能描述

该模块属于高频交互模块，应保证列表刷新、搜索返回和查看器跳转过程尽量平滑；搜索与查看采用“轻前端 + 远程数据 + 本地渲染桥接”的方式，避免在移动端执行重计算。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/recall.dart))

##### 8.1.3 输入

用户输入的自然语言查询、`model_assets` 中的资产记录、搜索接口返回的结果对象。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/recall.dart))

##### 8.1.4 输出

模型卡片列表、搜索结果列表、3D 查看页面。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/recall.dart))

##### 8.1.5 程序逻辑

程序逻辑为：页面初始化查询 `model_assets` → 用户输入搜索词 → 调用 `search-models` → 获取 `ply_path` 与位姿信息 → 进入 Viewer → 通过 JSBridge 加载模型。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/recall.dart))

【本轮新增 2026-03-09】建议补两个“细节逻辑事例”：

- 当 `ply_path` 为空时，Recall 会回退到本地 demo 模型路径，保证演示可继续；  
- 搜索结果若包含 `matched_frames`，页面切换为“结果 + 匹配帧横向列表”，点击帧可带 `transform_matrix` 进入 Viewer 做定点跳转。  
【代码依据】`app/lib/pages/recall.dart:419,432,683`

##### 8.1.6 限制条件

当前查看器主要面向 Android/iOS；模型实际可视化效果还依赖后端导出的文件格式与位姿文件完整性。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/webgl_viewer.dart))

#### 8.2 视频采集与任务创建模块

##### 8.2.1 功能描述

该模块负责调用手机摄像头录制视频，并在录制结束后跳转到信息提交页。它是后续视频转 3DGS 流程的前端入口。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/record.dart))

##### 8.2.2 性能描述

该模块需保证取景流畅、录制稳定，并对长录制时长进行限制，避免素材过大影响后续上传与处理。当前代码里已设置 180 秒上限。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/record.dart))

##### 8.2.3 输入

相机视频流、用户填写的视频名称。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/record.dart))

##### 8.2.4 输出

本地视频文件路径、视频缩略图、待提交的任务信息。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/record.dart))

##### 8.2.5 程序逻辑

程序逻辑为：用户点击录制 → 完成录制后生成缩略图 → 跳转 `VideoSubmitPage` → 用户填写名称并提交。需要注意的是，当前提交按钮仍只有 Toast，真正的“上传 + 写任务表”闭环建议写成“下一步联调目标”。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/record.dart))

【本轮新增 2026-03-09】再补三个“可靠性细节”可显著增强这一节：

1. App 切后台时若正在录制，会先停录再释放相机，避免相机资源泄漏和录制文件损坏；  
2. 录制超时 180 秒自动停录，避免超大文件压垮后续上传和处理；  
3. 录制期间隐藏底部导航，减少误触导致录制中断。  
【代码依据】`app/lib/pages/record.dart:62-82,135,150`

##### 8.2.6 限制条件

当前该模块的主要限制不是 UI，而是提交流程尚未闭环，因此不建议在正式稿里写成“视频任务创建已经全链路完成”。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/app/lib/pages/video_submit.dart))

#### 8.3 云端重建与资产入库模块

##### 8.3.1 功能描述

该模块由 Python Worker 实现，负责轮询任务、识别 `task_type`、调用对应 pipeline、压缩模型、上传结果并更新数据库。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/ai_engine/3dgs/src/core/worker.py))

##### 8.3.2 性能描述

该模块属于异步长任务模块，重点在于稳定性、可恢复性和输出兼容性，而不是前台即时返回。为此系统将日志、状态、质量评分和失败回退都写入任务表或资产表。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

##### 8.3.3 输入

`processing_tasks` 表中的任务记录，以及 `raw/video.mp4` 或 `raw/image.png`。([GitHub](https://github.com/tianxingleo/BrainDance/blob/tianxingleo-DLUT-L20/docs/API_DOC.md))

##### 8.3.4 输出

`point_cloud.ply/.splat/.ksplat`、`transforms.json`、`preview.jpg`、`model_assets` 记录以及最终任务状态。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/ai_engine/3dgs/src/core/worker.py))

##### 8.3.5 程序逻辑

程序逻辑为：轮询 `pending` 任务 → 更新为 `processing` → 下载素材 → 创建 pipeline → 运行 `video_3dgs` / `single_image_sam3d` / `single_image_sharp` / `da3_feed_forward_3dgs` → 压缩与上传模型 → upsert `model_assets` → 标记任务完成或失败。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/ai_engine/3dgs/src/core/worker.py))

【本轮新增 2026-03-09】补充“该流程里的关键工程优化点”：

- 任务类型分发由工厂统一管理，新增 pipeline 时不需要改 Worker 主流程；  
- AI 分析元数据（评分/标签/描述）在模型生成后立即同步回任务与资产表；  
- `webgl_poses.json` 上传时同步 image_url，支持 Recall 搜索结果中的关键帧可视化；  
- finally 阶段强制清理源文件和工作区，降低长期运行时磁盘占用。  
【代码依据】`ai_engine/3dgs/src/core/factory.py:9-20`；`ai_engine/3dgs/src/core/worker.py:267-299,447-460`；`ai_engine/3dgs/src/modules/spatial_anchor.py:152-173`

##### 8.3.6 限制条件

该模块依赖 Supabase 配置、对象存储路径规范、GPU 环境和外部模型仓库；对于新技术链路，尤其是 DA3 与 SHARP，建议保持“已接入能力、持续验证中”的文档口径。([GitHub](https://raw.githubusercontent.com/tianxingleo/BrainDance/tianxingleo-DLUT-L20/ai_engine/3dgs/src/core/worker.py))

------

这一版已经够你直接落进模板了。先把文中的 `<u>待确认</u>` 三处技术口径核一下，再定稿。

【本轮新增 2026-03-09】代码证据索引（答辩速查）

| 文档主题 | 核心证据 |
| --- | --- |
| Recall 查询/搜索/跳转 | `app/lib/pages/recall.dart:281,297,419,683` |
| WebGL 代理/缓存/证书绕过 | `app/lib/pages/webgl_viewer.dart:74,79,103,198,307,321` |
| 录制可靠性策略 | `app/lib/pages/record.dart:62-82,135,150,173,176` |
| 提交流程当前状态 | `app/lib/pages/video_submit.dart:52` |
| 生成页联调状态 | `app/lib/pages/generate.dart:512` |
| Worker 任务状态机 | `ai_engine/3dgs/src/core/worker.py:183,194,305,317,425` |
| 压缩与多格式交付 | `ai_engine/3dgs/src/utils/ply_utils.py:152-200` |
| 图像质量清洗（OpenCV） | `ai_engine/3dgs/src/modules/image_proc.py:20,44,51,64` |
| 坐标系转换（Three.js 列主序） | `ai_engine/3dgs/src/modules/spatial_anchor.py:210,212` |
| 搜索模型与向量模型 | `supabase/functions/search-models/index.ts:342,430` |

【本轮新增 2026-03-09】待你确认数据清单（填完即可定稿）

| 序号 | 需要确认的数据口径 | 当前文档状态 | 你需要给的最终值（建议格式） |
| --- | --- | --- | --- |
| 1 | 模型压缩收益（体积） | 已写“显著下降”，百分比待定 | 例如：`平均压缩率 88.4%（n=20）` |
| 2 | 模型加载耗时优化 | 已写“显著加速”，秒数待定 | 例如：`冷启动 18.7s->3.1s，热启动 6.2s->1.8s` |
| 3 | DA3 多图生成耗时 | 目前是描述性表述 | 例如：`1080p/120帧，端到端 2m14s` |
| 4 | 单图 SAM3D/SHARP 耗时 | 目前是描述性表述 | 例如：`SHARP 11.2s，SAM3D 14.6s（L20）` |
| 5 | 搜索响应时延 | 目前无具体数字 | 例如：`P50 420ms，P95 980ms（50次）` |
| 6 | 录制失败/重试率 | 目前无统计项 | 例如：`录制提交失败率 3.2%，重试成功率 91%` |

【本轮新增 2026-03-09】实测记录模板（可直接粘到附录）

| 日期 | 机型/服务器 | 网络 | 模型格式 | 文件大小 | 首次加载(s) | 二次加载(s) | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-09 | OPPO Find X8 + L20x2 | Wi-Fi 6 | `.ply` |  |  |  |  |
| 2026-03-09 | OPPO Find X8 + L20x2 | Wi-Fi 6 | `.splat` |  |  |  |  |
| 2026-03-09 | OPPO Find X8 + L20x2 | Wi-Fi 6 | `.ksplat` |  |  |  |  |

【本轮新增 2026-03-09】最小取数建议（10 分钟版本）

1. 从同一场景导出 `.ply/.splat/.ksplat` 三种格式，记录文件大小；  
2. 在同一台手机上各打开 3 次，分别记首次加载和二次加载平均值；  
3. 搜索词固定 10 条，统计请求耗时，计算 P50/P95；  
4. 把结果填入上表后，将正文中的“明显提升/显著下降”改成“具体数值+样本数”。  
【代码依据】`ai_engine/3dgs/src/core/worker.py:305,317`；`app/lib/pages/webgl_viewer.dart:177,307`；`supabase/functions/search-models/index.ts`
