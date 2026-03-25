# Agent 联调与回归清单

本文档用于把 `agent-recall` 从“接口可用”推进到“可联调、可回归、可稳定迭代”。

## 1. 当前入口

- 旧搜索入口：`/functions/v1/search-models`
- 新稳定入口：`/functions/v1/agent-recall`

两者当前已经共享 `supabase/functions/search-models/shared.ts` 中的搜索逻辑：

- 时间解析
- Embedding 生成
- `match_memory_poses` 向量检索
- 基础结果结构

这意味着后续调整阈值、时间解析或召回逻辑时，应优先修改共享层，而不是分别改多个入口。

## 2. agent-recall 请求与响应

请求：

```json
{
  "query": "上周拍的红色杯子"
}
```

响应：

```json
{
  "answer": "已找到最相关的空间记忆，命中场景 scene-demo，相关描述为“桌面上的红色杯子”，相似度约 91%。",
  "evidence": {
    "sceneId": "scene-demo",
    "similarity": 0.91,
    "matchedFrames": [
      {
        "imageName": "frame_0001.jpg",
        "similarity": 0.91,
        "transformMatrix": [1, 0, 0, 0]
      }
    ]
  },
  "actions": [
    {
      "type": "open_scene",
      "sceneId": "scene-demo",
      "modelId": "model-demo",
      "ply": "user-id/scene-demo/output/point_cloud.ply",
      "poses": null
    },
    {
      "type": "fly_to_pose",
      "sceneId": "scene-demo",
      "imageName": "frame_0001.jpg",
      "matrix": [1, 0, 0, 0]
    },
    {
      "type": "highlight_region",
      "sceneId": "scene-demo",
      "imageName": "frame_0001.jpg",
      "label": "红色杯子",
      "matrix": [1, 0, 0, 0]
    }
  ]
}
```

## 3. 前端动作映射表

当前 Viewer 联调应只接受以下稳定动作：

- `open_scene`
  - 含义：打开场景资源
  - 必要字段：`sceneId`
  - 常用附加字段：`modelId`、`ply`、`poses`
- `fly_to_pose`
  - 含义：飞到证据视角
  - 必要字段：`sceneId`
  - 常用附加字段：`imageName`、`matrix`
- `highlight_region`
  - 含义：高亮证据区域或热点
  - 必要字段：`sceneId`
  - 常用附加字段：`imageName`、`label`、`matrix`

前端不要再直接依赖旧命名：

- `open_model`
- `highlight_hotspot`

这些命名目前只允许存在于实验链路 `spatial-search-agent` 中。

## 4. 联调检查项

建议固定以下 4 条真实查询做人工联调：

1. `黑色耳机在哪`
2. `窗边那个台灯还在吗`
3. `上周拍的红色杯子`
4. `最像厨房角落堆着纸箱的空间`

每条查询至少检查：

1. 是否返回非空 `answer`
2. 是否返回 `evidence.sceneId`
3. 是否返回至少一个稳定动作
4. `fly_to_pose.matrix` 是否能驱动 Viewer 到达可信位置
5. `highlight_region` 的标签是否与实际命中内容一致

## 5. 回归分层

### 5.1 纯函数回归

适合每次改动都跑：

- `deno test supabase/functions/search-models/test.ts`
- `deno test supabase/functions/agent-recall/test.ts`

### 5.2 真实数据 smoke test

适合联调前或发布前跑：

- `deno test supabase/functions/agent-recall/smoke.ts`

该用例依赖：

- 本地或远程 Supabase 可访问
- `agent-recall` 已启动
- 环境变量中有 `SUPABASE_SERVICE_ROLE_KEY`

## 6. 当前限制

- `recall.dart` 仍消费旧的 `search-models` 列表结构，尚未切到 `agent-recall`。
- Viewer 侧动作消费链路还需要单独联调，不应在未完成 handler
  适配时直接替换线上入口。
- `spatial-search-agent` 仍保留为 LangChain 试验链路，不作为稳定前端入口。
