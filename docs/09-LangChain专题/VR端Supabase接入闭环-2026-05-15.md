# VR 端 Supabase 接入闭环 - 2026-05-15

## 背景

- VR 端原先已经有 Supabase client、登录面板和 HUD 入口，但登录后不会主动读取 `model_assets`。
- 桌面面板里的模型搜索主要过滤本地 payload / 静态 catalog，无法形成“登录 -> 查库 -> 生成模型 URL -> 切换查看”的云端闭环。
- 本轮目标是把 VR 端推进为独立 WebXR 客户端，先覆盖 Flutter 已有的鉴权、模型选择、模型查看和基础标记展示能力。

## 已完成

- `vr-3dgs-viewer/src/engine/supabaseClient.ts`
  - 兼容 `VITE_BD_SUPABASE_URL / VITE_BD_SUPABASE_ANON_KEY` 和 dashboard 使用的 `VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY`。
  - 保留 Supabase JS 自带的持久会话和自动刷新能力。
- `vr-3dgs-viewer/vite.config.ts`
  - 当 HTTPS dev server 访问 `http://` Supabase 本地地址时，默认把 `/supabase-proxy` 转发到 `VITE_BD_SUPABASE_URL`。
  - 避免未配置 `VITE_BD_SUPABASE_PROXY_TARGET` 时请求落回 Vite 前端 HTML，导致 Supabase SDK 报 `Unexpected token '<'`。
- `vr-3dgs-viewer/src/services/modelRepository.ts`
  - 新增 VR 端 Supabase 数据仓库层。
  - 支持从 `model_assets` 读取模型资产，按“我的模型 / 社区模型”两种来源查询。
  - 使用 Supabase Storage `createSignedUrl()` 解析 `ply_path`、`preview_img_path` 和推导出的 `webgl_poses.json`，避免依赖公开 bucket。
  - 支持从 `memory_poses` 读取当前模型的空间标记，并转换成 VR 端已有的 marker / search result 结构。
- `vr-3dgs-viewer/src/components/VrGaussianViewer.vue`
  - 登录成功后自动刷新云端模型列表，并默认加载第一个可用模型。
  - 同步已有 Supabase 会话后也会刷新模型列表，避免刷新页面后只停留在本地 catalog。
  - 桌面面板新增“刷新模型 / 我的模型 / 社区模型 / 云端刷新”入口。
  - VR HUD 的模型页新增“刷新云端”和来源切换按钮。
  - 切换模型时会尝试读取对应 `memory_poses`，刷新标记、证据和导航点展示。
- `vr-3dgs-viewer/src/types/viewer.ts` 与 `vr-3dgs-viewer/src/engine/bridge.ts`
  - `BrainDanceAuthSession` 不再保留 `accessToken / refreshToken` 字段。
  - VR 端本地 client state 只保存用户摘要信息，真正的 token 生命周期交给 Supabase Auth。

## 验证

- 已在 `vr-3dgs-viewer` 执行：

```bash
npm run build
```

- 结果：`vue-tsc --build` 与 `vite build` 均通过。
- Vite 输出了主 chunk 超过 500 kB 的提示，这是 Three.js / 3DGS viewer 打包体积带来的既有风险，不影响本次功能闭环。
- 已补充本地 HTTP Supabase 代理兜底，避免 `memory_poses` 查询收到 Vite `index.html`。

## 当前限制

- 本轮没有直接接入 `agent-recall` 的流式 Recall 搜索，VR 端搜索框当前用于云端 `model_assets` 文本过滤和本地列表过滤。
- `memory_poses.pose_data` 的字段兼容了 `matrix / transform_matrix / transform / camera_to_world / position / translation / location`，如果线上数据格式继续分叉，需要在 repository 层继续补 schema 兼容。
- 社区模型来源当前按 `model_assets` 读取，若后续要严格复刻 Flutter 社区页，需要继续接 `community_posts` 的发布状态、封面、点赞和权限字段。

## 下一步建议

- 把 `agent-recall` 的 SSE / NDJSON 消费封装成 VR 端 service，把 Flutter Recall 的 `top_candidates / tool_trace / follow_up / session_state` 映射到 HUD 结果和空间跳转。
- 对 VR 端增加最小 E2E 调试脚本，覆盖登录、刷新模型、加载模型、读取 `memory_poses` 四个关键节点。
- 如果线上 bucket 已启用严格 RLS，需要确认浏览器端 authenticated 用户可对自己的 Storage 路径调用 `createSignedUrl()`。
