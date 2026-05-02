# 时间切片（Time Peeling）设计与实现

## 1. 概述

**Time Peeling（时间切片）** 是 BrainDance 的核心交互特性之一，允许用户在同一场景的多个不同时刻拍摄的 3DGS 模型之间自由切换浏览。用户可以通过时间轴直观地查看场景随时间的变化过程。

**核心能力：**
- 按场景名称对模型分组，每组内按拍摄时间排列
- 在 WebView 3D 查看器中无需退出即可切换模型
- 本地缓存已下载的模型，实现秒级切换
- Recall 页面以水平时间轴轮播卡片展示各组模型

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Recall 页面                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  TimePeelingList (Flutter)                      │    │
│  │  ├── 按场景名分组（_groupModelsByName）          │    │
│  │  ├── 每组一个 _TimePeelingSlot（水平轮播）       │    │
│  │  └── 底部 _TimelinePainter（时间轴节点）         │    │
│  └─────────────────────┬───────────────────────────┘    │
│                        │ 点击模型                        │
│                        ▼                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │  WebGLViewer (Flutter WebView)                  │    │
│  │  ├── _sendTimePeelingList() → JS Bridge         │    │
│  │  ├── _handleSwitchModel() ← JS Bridge           │    │
│  │  └── 本地代理 / 下载缓存                         │    │
│  └─────────────────────┬───────────────────────────┘    │
└────────────────────────┬────────────────────────────────┘
                         │ JS Bridge
                         ▼
┌─────────────────────────────────────────────────────────┐
│              GaussianViewer.vue (WebView)                │
│  ┌─────────────────────────────────────────────────┐    │
│  │  window.setModelListForTimePeeling(list, id)     │    │
│  │  ├── 接收模型列表，存入 modelList ref            │    │
│  │  └── 标记当前活动模型 activeModelId              │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  BottomSelector.vue                             │    │
│  │  ├── "空间"/"时间" 双 Tab 切换                   │    │
│  │  ├── 时间 Tab：缩略图列表（newest→oldest）       │    │
│  │  └── 点击 → emit('selectModel') → 切换           │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  onTimePeelingSelect(model)                     │    │
│  │  └── BrainDanceChannel.postMessage(switchModel)  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 数据流

### 3.1 模型列表加载（Flutter → WebView）

```
Supabase Storage
    │
    ▼
Recall 页面加载模型列表 (_allModels)
    │
    ├── 搜索/筛选 → _models
    │
    ▼
_groupModelsByName(_models)
    │ 按 display_name 分组
    │ 组内按 created_at 降序
    ▼
TimePeelingList / TimePeelingSlot 展示
    │
    ▼ 用户点击某模型
WebGLViewer 打开
    │
    ▼ 页面初始化后
_sendTimePeelingList()
    │ 构建 [{id, name, ply, poses, previewImg, createdAt}, ...]
    │ 通过本地代理 URL 提供 PLY / 预览图
    ▼
controller.runJavaScript("window.setModelListForTimePeeling(json, currentId)")
    │
    ▼
WebView 侧 modelList.value = list
BottomSelector.vue 渲染缩略图列表
```

### 3.2 模型切换（WebView → Flutter → WebView）

```
用户在 BottomSelector 中点击模型缩略图
    │
    ▼
onTimePeelingSelect(model)
    │ 更新 activeModelId
    ▼
BrainDanceChannel.postMessage({ action: 'switchModel', modelId })
    │
    ▼ JS Bridge 回调
_handleSwitchModel(data)
    │ 1. 在 timePeelingModels 中找到目标模型
    │ 2. 从 Supabase Storage 生成公开 URL
    │ 3. 检查本地缓存 (ApplicationDocumentsDirectory)
    │    ├── 已缓存 → 直接用本地文件
    │    └── 未缓存 → 下载到本地，存为 .tmp 后 rename
    │ 4. 通过本地代理提供文件 URL
    ▼
controller.runJavaScript("window.loadModelFromFlutter({ply, poses})")
    │
    ▼
WebGL 加载新模型，场景无缝切换
```

---

## 4. 关键文件与代码

| 文件 | 角色 |
|------|------|
| `app/lib/pages/recall/recall_view.dart` | Recall 页面，根据条件选择展示 `TimePeelingList` 或 `RecallModelGrid` |
| `app/lib/pages/recall/recall_data_sync.dart` | `_groupModelsByName()` 模型分组逻辑 |
| `app/lib/pages/recall/model_grid.dart` | `TimePeelingList`、`_TimePeelingSlot`、`_TimelinePainter` 组件 |
| `app/lib/pages/webgl_viewer.dart` | `_sendTimePeelingList()`、`_handleSwitchModel()`、`_findCurrentModelId()` |
| `app/lib/pages/time_peeling.dart` | 独立页面占位（coming soon） |
| `3dgs_viewer/.../GaussianViewer.vue` | `setModelListForTimePeeling()`、`onTimePeelingSelect()`、`loadModelFromFlutter()` |
| `3dgs_viewer/.../BottomSelector.vue` | 底部选择器 UI（空间/时间双 Tab、缩略图滚动列表） |

---

## 5. Flutter 侧实现细节

### 5.1 模型分组 — `_groupModelsByName()`

**文件：** `app/lib/pages/recall/recall_data_sync.dart`

- 以 `display_name`（或 `tags`、`scene_id`）作为分组键
- 组内按 `created_at` **降序**排列（最新在前）
- 返回 `Map<String, List<Map<String, dynamic>>>`

### 5.2 展示组件 — `TimePeelingList`

**文件：** `app/lib/pages/recall/model_grid.dart`

**TimePeelingList（外层）：**
- 接收 `groupedModels`，将分组按键排序（最新组在前）
- 每组渲染一个 `_TimePeelingSlot`

**_TimePeelingSlot（每组）：**
- 顶部：场景名称 + 模型数量徽章
- 中部：`PageView.builder` 水平轮播
  - `viewportFraction: 0.52`，中心放大效果
  - index 0 为「新建任务」卡片（"+" 按钮）
  - index 1+ 为模型卡片（`RecallModelTile`，`imageOnly: true`）
  - 非选中项 `scale: 0.82`、`opacity: 0.5` 渐变
  - `ShaderMask` 实现左右边缘渐隐
- 底部：`_TimelinePainter` 自绘时间轴
  - 连接线 + 圆形节点（选中节点更大更亮）
  - 选中节点下方显示时间标签（`MM/DD HH:mm`）

### 5.3 模型列表发送 — `_sendTimePeelingList()`

**文件：** `app/lib/pages/webgl_viewer.dart`

**处理流程：**
1. **空列表兜底：** 若 `timePeelingModels` 为空，构造单元素列表（使用当前模型 URL）
2. **构建模型列表：** 遍历 `timePeelingModels`，为每个模型生成：
   - `ply`：PLY 文件的本地代理 URL（`http://127.0.0.1:{port}/proxy/{encodedUrl}`）
   - `poses`：`webgl_poses.json` 的公开 URL
   - `previewImg`：预览图的本地代理 URL
   - `createdAt`：创建时间
3. **发送到 WebView：** `controller.runJavaScript("window.setModelListForTimePeeling(json, currentId)")`

### 5.4 模型切换 — `_handleSwitchModel()`

**文件：** `app/lib/pages/webgl_viewer.dart`

**处理流程：**
1. 在 `timePeelingModels` 中按 `modelId` 查找目标模型
2. 生成 Supabase 公开 URL
3. **本地缓存检查：**
   - 缓存路径：`ApplicationDocumentsDirectory/{sanitized_path}`
   - 命中 → 直接使用 `http://127.0.0.1:{port}/local_models/{encodedPath}`
4. **未命中则下载：**
   - HTTP 下载到 `.tmp` 文件
   - 下载完成后 rename 为正式文件名
   - 下载失败则 fallback 到代理 URL
5. 构建 `{ply, poses}` JSON，调用 `window.loadModelFromFlutter(payload)`

---

## 6. WebView 侧实现细节

### 6.1 模型列表接收

**文件：** `GaussianViewer.vue`

```javascript
window.setModelListForTimePeeling = (list, currentId) => {
  modelList.value = list;
  activeModelId.value = currentId || list[0]?.id || '';
};
```

### 6.2 BottomSelector 组件

**文件：** `BottomSelector.vue`

- **双 Tab 模式：** "空间"（视角切换）和 "时间"（模型切换）
- 仅当 `hasModels`（模型数 > 1）时显示 Tab 切换按钮
- 时间模式下展示 `sortedModels`（按 `createdAt` 降序）
- 缩略图支持拖拽滚动
- 选中项有放大高亮效果
- 每个缩略图底部显示 `MM/DD HH:mm` 时间标签

### 6.3 模型选中回调

```javascript
function onTimePeelingSelect(model) {
  activeModelId.value = model.id;
  window.BrainDanceChannel.postMessage(
    JSON.stringify({ action: 'switchModel', modelId: model.id })
  );
}
```

---

## 7. 本地代理与缓存机制

Time Peeling 的流畅切换依赖于本地 HTTP 代理：

```
Flutter 侧启动本地 HTTP Server (127.0.0.1:{port})
    │
    ├── /proxy/{encodedUrl}     → 转发远程请求（绕过 CORS）
    │
    └── /local_models/{path}    → 读取本地缓存文件
```

**缓存策略：**
- 首次加载：通过代理直接从 Supabase 下载
- 切换模型时：检查本地缓存，命中则直接读取，未命中则先下载再缓存
- 缓存路径：`{ApplicationDocumentsDirectory}/{sanitized_url_path}`

---

## 8. UI 交互总结

| 场景 | 触发位置 | 行为 |
|------|----------|------|
| Recall 页面浏览 | TimePeelingSlot 轮播 | 水平滑动查看不同时刻的模型缩略图 |
| 进入 3D 查看器 | 点击缩略图 | 打开 WebGLViewer，加载模型 + 发送 TimePeeling 列表 |
| 查看器内切换 | BottomSelector 时间 Tab | 点击缩略图 → Flutter 下载/缓存 → WebView 加载新模型 |
| 新建任务 | TimePeelingSlot 首位 "+" 卡片 | 跳转到拍摄页 |
| 模型操作 | 长按缩略图 | 弹出操作菜单（查看详情、重命名、下载、删除、分享） |
