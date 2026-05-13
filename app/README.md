# BrainDance App

`app/` 是 BrainDance 的 Flutter 客户端，负责移动端采集、任务提交、结果查看、Recall 检索与本地 AI 入口。

## 当前状态

当前代码已经包含以下主链路：

- 登录与会话管理
- 相机录制与本地素材选择
- 视频上传与任务创建
- 任务列表与状态更新
- Recall 资产页
- Recall 本地 AI 问答入口
- Community 社区流与地图探索页
- 基于 WebView 的移动端 3D 模型查看
- 设置页与本地配置持久化

对应页面主要位于：

- `lib/pages/login.dart`
- `lib/pages/record.dart`
- `lib/pages/video_submit.dart`
- `lib/pages/task_list.dart`
- `lib/pages/recall.dart`
- `lib/pages/recall/local_ai_panel.dart`
- `lib/pages/community.dart`
- `lib/pages/webgl_viewer.dart`
- `lib/pages/settings.dart`

本地 AI 相关服务位于：

- `lib/services/local_rag_index.dart`
- `lib/services/local_text_embedder.dart`

## 依赖与环境

- Flutter SDK `3.10.x` 对应的 Dart SDK
- Android Studio / Xcode
- 可用的 Supabase 项目或本地 Supabase 环境

项目当前使用的关键依赖见 `pubspec.yaml`：

- `supabase_flutter`
- `camera`
- `image_picker`
- `webview_flutter`
- `flutter_riverpod`
- `flutter_dotenv`
- `tdesign_flutter`
- `dio`
- `sqflite`
- `llamadart`

## Recall 本地 AI 现状

当前分支已经恢复 Recall 的本地 AI 入口，接入方式不是云端推理，而是：

- 本地模型格式：`GGUF`
- 本地推理引擎：`llamadart`
- 本地检索索引：`sqflite + hashing embedder`
- 模型分发：面板中填写下载链接后下载到应用私有目录
- 默认模型文件名与对象路径仍指向 `qwen3-1.7b-braindance-q5-k-m-imatrix.gguf`

当前代码现状可以概括为：

1. `recall.dart` 负责页面与模式切换
2. `local_ai_panel.dart` 负责模型下载、加载与状态展示
3. `local_rag_index.dart` 负责本地 Recall 资产索引、缓存与相似度搜索

这意味着当前最顺的模型接入方式是 GGUF，而不是直接加载 Hugging Face LoRA adapter。

## 环境变量

先复制环境变量模板：

```bash
cd app
cp .env.example .env
```

然后填写：

```env
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY
```

## 本地运行

```bash
cd app
flutter pub get
flutter run
```

如果需要查看可连接设备：

```bash
flutter devices
```

## 目录说明

- `lib/main.dart`：应用入口、Supabase 初始化、路由注册
- `lib/configs/`：应用配置、Supabase 配置、录制与生成配置
- `lib/pages/`：主要页面
- `lib/services/`：任务通知、本地 AI、Recall 本地索引等服务
- `assets/webgl/`：内置 WebGL 资源

## 当前约定

- App 通过 `.env` 读取 `SUPABASE_URL` 和 `SUPABASE_ANON_KEY`
- 任务提交默认写入 `processing_tasks`，列表页优先使用 `display_name`
- 视频与缩略图默认上传到 `braindance-assets/{user_id}/{scene_id}/raw/`
- Recall 页从 `model_assets` 读取结果，并尝试推导对应的模型与位姿文件 URL
- Recall 本地 AI 面板默认围绕 `GGUF + llamadart` 链路工作，默认参数已经收紧为更偏移动端的上下文、batch 和输出长度配置
- Community 页从 `community_posts` 读取贴文，并通过关联的 `model_assets` 还原模型地址与封面

## 说明

这份 README 只保留客户端本身的运行和结构说明。完整系统链路请回到项目根目录的 [README.md](/ltx-data/BrainDance/README.md)。

补充说明：

- 本轮仅补齐文档，没有在当前机器上执行 `flutter analyze`、`flutter test` 或真机构建
- 原因是当前环境中 `flutter` / `dart` CLI 不在 PATH，无法在这一轮做客户端构建校验
