# VR 默认模型目录

本目录用于本地开发 fallback。VR Viewer 没有收到 `payload` 时会尝试加载：

- `scene_auto_sync_raw.ply`
- `webgl_poses.json`
- `vr_config.json`

仓库不提交大型 3DGS 模型文件。调试时可以把本地模型临时放到这里，或通过 URL payload 指向 Supabase Storage / 本地代理地址。
