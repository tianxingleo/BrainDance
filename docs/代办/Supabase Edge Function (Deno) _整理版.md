# Supabase Edge Function (Deno) 替代 Python HTTP 接口方案

## 概述

使用 Supabase Edge Function (Deno) 取代 Python 的 HTTP 业务接口部分（`server.py`），是实现 **Serverless 架构** 的最后一块拼图。

这样做的最大好处是：

1. **彻底消灭 `server.py`**：Python 项目不再需要运行 HTTP 服务（FastAPI/Flask），也不需要暴露端口。
2. **职责分离**：
   - **Python Worker**：只负责**干苦力**（GPU 渲染、3DGS 训练），变成一个纯粹的后台消费者，藏在内网里。
   - **Edge Function**：负责**脑力活**（RAG 搜索、鉴权、调用 LLM API），直接响应前端请求。
3. **安全性提升**：API Key (DashScope/OpenAI) 存在 Supabase Secrets 里，既不暴露给前端，也不需要硬编码在 Python 代码里。

---

## 新架构：Edge Function 负责搜索

在这种架构下，Python 代码只剩下 `worker.py`。前端 Flutter 的搜索请求直接打给 Supabase 云端。

### 架构图变化

```
sequenceDiagram
    participant App as 📱 Flutter
    participant Edge as ⚡ Edge Function (Deno)
    participant LLM as ☁️ DashScope API
    participant DB as 🐘 Supabase DB
    participant Py as 🐍 Python Worker

    Note over App, Py: 场景一：搜索 (RAG) - Python 不参与
    App->>Edge: 1. POST /functions/v1/search {query: "红色的杯子"}
    Edge->>LLM: 2. 请求 Embedding
    LLM-->>Edge: 3. 返回向量
    Edge->>DB: 4. rpc('match_model_assets', vector)
    DB-->>Edge: 5. 返回模型列表
    Edge-->>App: 6. 返回 JSON 结果

    Note over App, Py: 场景二：生成 (3DGS)
    App->>DB: 1. 上传视频 & 插入任务
    Py->>DB: 2. 轮询/监听任务
    Py->>Py: 3. GPU 疯狂计算...
    Py->>DB: 4. 更新结果
```

---

## 代码实现 (TypeScript/Deno)

需要创建一个新的 Edge Function。

**步骤 1: 初始化**

```bash
supabase functions new search
```

**步骤 2: 编写代码 (`supabase/functions/search/index.ts`)**

这是调用阿里云百炼 (DashScope) 并查库的完整逻辑。

```typescript
// Setup:
// 1. 在 Supabase Dashboard -> Settings -> Functions 填入 DASHSCOPE_API_KEY
// 2. 部署: supabase functions deploy search --no-verify-jwt

import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  // 1. 处理 CORS (允许 Flutter 跨域调用)
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { query } = await req.json()
    if (!query) throw new Error("缺少搜索关键词 query")

    // 2. 调用 DashScope (阿里云) 获取 Embedding
    // 注意：这里手动 fetch，比装 SDK 更轻量
    const apiKey = Deno.env.get('DASHSCOPE_API_KEY')
    const embeddingResp = await fetch('https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'text-embedding-v1', // 确保和你 Python 入库时用的模型一致
        input: { texts: [query] }
      })
    })

    const embeddingData = await embeddingResp.json()
    
    if (!embeddingData.output || !embeddingData.output.embeddings) {
        console.error("Embedding Error:", embeddingData)
        throw new Error("AI 向量生成失败")
    }

    const vector = embeddingData.output.embeddings[0].embedding

    // 3. 连接 Supabase (使用 Service Role 调用 RPC)
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? ''
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    const supabase = createClient(supabaseUrl, supabaseKey)

    // 4. 执行向量搜索
    const { data: results, error } = await supabase.rpc('match_model_assets', {
      query_embedding: vector,
      match_threshold: 0.4, // 阈值
      match_count: 10       // 返回数量
    })

    if (error) throw error

    // 5. 拼接完整图片/模型 URL (可选)
    const formattedResults = results.map((item: any) => ({
        ...item,
        model_url: `${supabaseUrl}/storage/v1/object/public/braindance-assets/${item.ply_path}`
    }))

    return new Response(JSON.stringify({ code: 200, data: formattedResults }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    })

  } catch (error) {
    return new Response(JSON.stringify({ code: 500, error: error.message }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    })
  }
})
```

---

## Flutter 端调用修改

前端不再调用 `http://127.0.0.1:8000/search`，而是直接调用 Supabase Function。

```dart
// Flutter 代码
final response = await Supabase.instance.client.functions.invoke(
  'search',
  body: {'query': '红色的杯子'},
);

if (response.status == 200) {
  final list = response.data['data'];
  // 渲染列表...
}
```

---

## 方案优势

1. **冷启动快**：Deno Edge Function 几乎是秒起的，而 Python 容器如果休眠了，唤醒很慢。
2. **省钱**：Edge Function 按调用次数收费（且有免费额度），不需要为了一个偶尔用一下的搜索功能，挂着一台 24 小时运行的 Python 服务器。
3. **Python 专注化**：Python 项目可以完全砍掉 `FastAPI/Flask` 依赖，砍掉 `uvicorn`，砍掉 `server.py`。它变成了一个纯粹的、离线的、只跟显卡打交道的计算脚本。

---

## 结论

采用这个方案后，"All-in-Supabase" 就非常彻底了：

- **Web 服务层**: Supabase (Auth + DB + Realtime + Edge Functions)
- **计算层**: Python (GPU Worker)
- **端侧**: Flutter

这是最优雅的现代 AI 应用架构。

---

---

## 补充：使用 Edge Function 替代 Go/Python 中间件

在 Supabase 生态中，**使用 Edge Function (Deno) 是比使用 Go 更"原生"、更轻量、更推荐的做法**。

针对 **BrainDance** 项目，可以完全用 Supabase Edge Functions 替代掉原本可能需要的 Go (或 Python FastAPI) **中间件层**。

### 1. 为什么要用 Edge Function 替代 Go？

原本 Go/Python 中间件的作用主要是：**接收前端搜索请求 -> 调用 LLM 拿向量 -> 查数据库 -> 返回结果**。

| 对比维度 | Go / Python (独立后端) | Supabase Edge Function (Deno) | 结论 |
| :--- | :--- | :--- | :--- |
| **部署维护** | 需要买服务器/容器，自己运维，自己管 SSL | **Serverless**，随 Supabase 部署，0 运维 | ✅ Deno 胜 |
| **网络延迟** | 前端 -> 你的服务器 -> Supabase (多一跳) | 前端 -> Edge Function (部署在全球边缘) -> Supabase (内网级速度) | ✅ Deno 胜 |
| **开发成本** | 需要配置路由、CORS、数据库连接池 | 开箱即用，自带 Supabase Client，自动处理 Auth | ✅ Deno 胜 |
| **计费** | 24小时开机费 | **按调用次数收费** (没流量不花钱) | ✅ Deno 胜 |
| **性能** | 高并发下 Go 更有优势 | V8 引擎 (JS/TS) 足够处理 API 胶水逻辑 | 🟢 够用 |

**结论**：对于 3DGS 训练（重计算），必须用 Python；但对于 **API 接口（如搜索、鉴权、支付回调）**，Edge Function 是完美替代品。

---

### 2. 实战教程

要把之前规划的 `server.py` (FastAPI 搜索服务) 废弃掉，改用 **Edge Function** 实现。

#### 第一步：初始化 Function

在项目根目录下：

```bash
# 创建一个名为 "search-models" 的函数
supabase functions new search-models
```

这会在 `supabase/functions/search-models/index.ts` 生成一个文件。

#### 第二步：编写 Deno 代码

Edge Function 使用 Deno (TypeScript) 运行。需要在这里做两件事：调用阿里云/OpenAI 获取 Embedding，然后调用数据库 RPC。

编辑 `supabase/functions/search-models/index.ts`：

```typescript
// 引入依赖 (Deno 直接从 URL 引入，无需 npm install)
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import OpenAI from 'https://esm.sh/openai@4'

// 设置 CORS 头 (允许 Flutter 跨域调用)
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  // 1. 处理 CORS 预检请求
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // 2. 获取请求参数
    const { query } = await req.json()
    if (!query) throw new Error('缺少搜索关键词 q')

    // 3. 初始化 OpenAI Client (用于调用 DashScope/Qwen)
    // 注意：Edge Function 会自动读取后台配置的密钥，无需本地 .env
    const apiKey = Deno.env.get('DASHSCOPE_API_KEY')
    const openai = new OpenAI({
      apiKey: apiKey,
      baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    })

    // 4. 初始化 Supabase Client
    // Deno.env.get('SUPABASE_URL') 等是系统自动注入的
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      // 关键：复用前端传来的 Auth Token，确保 RLS 权限生效 (如果你的表是私有的)
      { global: { headers: { Authorization: req.headers.get('Authorization')! } } }
    )

    // 5. 生成 Embedding (调用 AI)
    console.log(`正在搜索: ${query}`)
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-v2", // 确保和入库时模型一致
      input: query,
    })
    const embedding = embeddingResponse.data[0].embedding

    // 6. 调用数据库 RPC 进行搜索
    const { data: documents, error } = await supabaseClient.rpc('match_model_assets', {
      query_embedding: embedding,
      match_threshold: 0.4, // 相似度阈值
      match_count: 10,      // 返回数量
    })

    if (error) throw error

    // 7. 返回结果
    return new Response(JSON.stringify({ code: 200, data: documents }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    })

  } catch (error) {
    return new Response(JSON.stringify({ code: 500, error: error.message }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    })
  }
})
```

#### 第三步：设置环境变量

Function 需要知道 `DASHSCOPE_API_KEY`。

1. 新建一个 `.env.local` 文件（如果还没有），或者直接在根目录 `.env` 里加：

   ```
   DASHSCOPE_API_KEY=sk-xxxxxx
   ```

2. **推送到 Supabase (本地开发不需要这一步，部署上线才需要)**：

   ```bash
   supabase secrets set --env-file .env
   ```

#### 第四步：本地运行与测试

确保 Supabase 还在运行 (`supabase start`)。

```bash
# 启动 Function 调试服务
supabase functions serve
```

现在可以用 curl 或者 Postman 测试这个接口：

- **URL**: `http://127.0.0.1:54321/functions/v1/search-models`
- **Method**: `POST`
- **Header**: `Authorization: Bearer <你的Anon Key>`
- **Body**: `{"query": "红色的杯子"}`

---

### 3. Flutter 前端调用

前端不需要调 `localhost:8000` 了，而是直接调 Supabase 的 Function。代码会变得非常简洁：

```dart
import 'package:supabase_flutter/supabase_flutter.dart';

final supabase = Supabase.instance.client;

Future<List> searchModels(String query) async {
  try {
    // 直接调用 Edge Function
    final response = await supabase.functions.invoke(
      'search-models', // 函数名
      body: {'query': query},
    );
    
    final data = response.data;
    if (data['code'] == 200) {
      return data['data']; // 返回搜索结果列表
    } else {
      print('Search Error: ${data['error']}');
      return [];
    }
  } catch (e) {
    print('Network Error: $e');
    return [];
  }
}
```

---

### 4. 架构图更新

架构非常清晰，没有多余的"中间商"：

- **重计算 (3DGS 训练)**: 🐢 Python Worker (监听 DB -> 显卡狂转 -> 存回 DB)
- **轻计算 (搜索/鉴权)**: 🐇 **Supabase Edge Function (Deno)**
- **数据存储**: 🗄️ Supabase Postgres + Vector
- **前端**: 📱 Flutter

---

### 5. 总结

1. **能代替 Go 吗？** 对于 API 接口层，**完全可以**，而且是绝配。
2. **优势**：代码量少（不用配路由、服务器），前端调用方便（直接集成在 SDK 里），部署简单（`supabase functions deploy` 一行命令）。
3. **唯一限制**：Edge Function 有执行时间限制（通常 60秒），但对于 Embedding + 搜索这种 1秒内完成的任务，完全没问题。切记**不要**把 3D 训练逻辑放到这里面就行。
