# 3dgs_viewer

`3dgs_viewer/` 是 BrainDance 的 Web 查看器与相关辅助脚本目录，用于把 3DGS 结果整理成浏览器可用的资源，并在本地进行查看和调试。

## 当前目录内容

- `my-3dgs-viewer/`：Vue 3 + Three.js 查看器前端
- `run_glomap.py`：本地视频到查看器资源的实验脚本
- `export_poses.py`：位姿导出工具
- `sync_images.py`：同步参考图片到查看器目录
- `tag_poses.py`：给位姿数据打中文标签
- `evaluate_poses.py` / `fix_poses.py` / `calc_transform.py`：位姿调试工具
- `add_tags.py`：临时辅助脚本，带硬编码路径，不建议直接当正式工具使用

## 当前定位

这个目录更接近“查看与调试工具箱”，不是主处理链路。真正的任务调度和模型生成仍然以 [ai_engine/3dgs/README.md](/home/ltx/projects/BrainDance/ai_engine/3dgs/README.md) 为准。

## 最常见的使用方式

### 1. 启动查看器前端

```bash
cd 3dgs_viewer/my-3dgs-viewer
npm install
npm run dev
```

### 2. 同步图片

```bash
cd 3dgs_viewer
python sync_images.py
```

这个脚本会尝试从 `3dgs_viewer/outputs/` 下最近的训练结果中找到 `transforms.json`，并把对应图片同步到 `my-3dgs-viewer/public/models/images/`。

### 3. 给位姿打标签

```bash
cd 3dgs_viewer
export DASHSCOPE_API_KEY=YOUR_API_KEY
python tag_poses.py
```

`tag_poses.py` 会读取：

- `my-3dgs-viewer/public/models/webgl_poses.json`

并输出：

- `my-3dgs-viewer/public/models/webgl_poses_with_tags.json`

## 说明

- `run_glomap.py` 仍然是实验性质较强的本地脚本，依赖较多，也带有一些硬编码参数
- 如果你只想看模型，优先进入 `my-3dgs-viewer/`
- 如果你要修改主链路中的 WebGL 查看体验，还需要同步关注 `app/assets/webgl/` 和移动端 WebView 接入方式
