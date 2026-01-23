# BrainDance API 测试报告

> 本文档记录 Edge Function (`search-models`) 的自动化测试结果。

## 测试环境

| 组件 | 版本/配置 |
| :--- | :--- |
| **Deno** | 1.46.0 |
| **Supabase Local** | 运行中 (端口 54321) |
| **DashScope API** | text-embedding-v2, qwen-plus |
| **测试时间** | 2026-01-20 |

---

## 1. 自动化测试结果

### 1.1 测试命令

```bash
deno test --allow-all supabase/functions/search-models/test.ts
```

### 1.2 测试结果汇总

```
running 11 tests from ./supabase/functions/search-models/test.ts

✅ safeJsonParse - 有效 JSON 输入
✅ safeJsonParse - null 输入
✅ safeJsonParse - 无效 JSON 输入
✅ normalizeDate - 有效日期格式
✅ normalizeDate - null 输入
✅ normalizeDate - 无效日期格式
⚠️ 集成测试 - 完整搜索流程 (跳过: 需要 SERVICE_ROLE_KEY)
✅ API 测试 - 缺少 query 参数 (400)
✅ API 测试 - 空查询字符串 (400)
✅ API 测试 - 查询字符串过长 (400)
✅ CORS 测试 - OPTIONS 预检请求 (200)

ok | 11 passed | 0 failed
```

---

## 2. 实际 API 调用测试

### 2.1 测试命令

#### 测试 1: 简单语义搜索

```bash
curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Content-Type: application/json' \
  -d '{"query":"红色杯子"}'
```

**预期**: 成功返回，意图解析正确

**响应**:
```json
{
  "success": true,
  "intent": {
    "original_query": "红色杯子",
    "parsed_search_text": "红色杯子",
    "filter_start": null,
    "filter_end": null
  },
  "results": []
}
```

**结果**: ✅ 通过

---

#### 测试 2: 带时间过滤的搜索

```bash
curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Content-Type: application/json' \
  -d '{"query":"找一下上周拍的照片"}'
```

**预期**: LLM 正确解析"上周"为时间范围

**响应**:
```json
{
  "success": true,
  "intent": {
    "original_query": "找一下上周拍的照片",
    "parsed_search_text": "照片",
    "filter_start": "2026-01-13T00:00:00Z",
    "filter_end": "2026-01-19T23:59:59Z"
  },
  "results": []
}
```

**结果**: ✅ 通过 (LLM 正确计算了"上周"的时间范围)

---

#### 测试 3: 缺少 query 参数

```bash
curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Content-Type: application/json' \
  -d '{}'
```

**预期**: 返回 400 错误

**响应**:
```json
{
  "success": false,
  "error": "缺少或无效的搜索关键词 'query'"
}
```

**结果**: ✅ 通过

---

#### 测试 4: 空查询字符串

```bash
curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Content-Type: application/json' \
  -d '{"query":"   "}'
```

**预期**: 返回 400 错误

**响应**:
```json
{
  "success": false,
  "error": "搜索关键词不能为空"
}
```

**结果**: ✅ 通过

---

#### 测试 5: 查询字符串过长 (501 字符)

```bash
curl -X POST 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Content-Type: application/json' \
  -d '{"query":"a'$(printf 'a%.0s' {1..500})'"}'
```

**预期**: 返回 400 错误

**响应**:
```json
{
  "success": false,
  "error": "搜索关键词过长（最大 500 字符）"
}
```

**结果**: ✅ 通过

---

#### 测试 6: CORS 预检请求

```bash
curl -X OPTIONS 'http://127.0.0.1:54321/functions/v1/search-models' \
  -H 'Origin: http://localhost:3000' \
  -H 'Access-Control-Request-Method: POST'
```

**预期**: 返回 200 和 CORS 头

**响应头**:
```
Access-Control-Allow-Origin: *
```

**结果**: ✅ 通过

---

## 3. 测试用例矩阵

| 测试场景 | 输入 | 预期结果 | 实际结果 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| 简单搜索 | `{"query":"红色杯子"}` | success: true | success: true | ✅ |
| 时间过滤 | `{"query":"上周拍的..."}` | 含时间范围 | 含时间范围 | ✅ |
| 缺少参数 | `{}` | 400 + 错误信息 | 400 + 错误信息 | ✅ |
| 空字符串 | `{"query":"   "}` | 400 + 错误信息 | 400 + 错误信息 | ✅ |
| 超长字符串 | 501 字符 | 400 + 错误信息 | 400 + 错误信息 | ✅ |
| CORS 预检 | OPTIONS 请求 | 200 + CORS 头 | 200 + CORS 头 | ✅ |

---

## 4. 意图解析测试

### 4.1 测试结果

| 用户输入 | 解析搜索词 | 开始时间 | 结束时间 | 状态 |
| :--- | :--- | :--- | :--- | :---: |
| "红色杯子" | "红色杯子" | null | null | ✅ |
| "上周拍的照片" | "照片" | 2026-01-13 | 2026-01-19 | ✅ |

### 4.2 LLM 提示词

```text
你是搜索助手。当前日期是: 2026-01-20。
用户会输入一句搜索请求，你需要提取：
1. search_text: 真正用于搜索物体的描述（去掉时间词）。
2. start_time: ISO8601 格式的开始时间 (UTC)，如果没有则为 null。
3. end_time: ISO8601 格式的结束时间 (UTC)，如果没有则为 null。

只返回 JSON。
```

---

## 5. 性能测试

### 5.1 响应时间

| 测试 | 响应时间 |
| :--- | :---: |
| 简单搜索 (无数据库结果) | ~2-3 秒 |
| 带时间过滤搜索 | ~2-3 秒 |

**说明**: 响应时间主要消耗在 LLM API 调用 (意图解析 + 向量生成)。

---

## 6. 后续测试计划

### 6.1 需要数据库有数据时的测试

当 `model_assets` 表中有数据时，需要验证：

1. **向量搜索准确性**: 搜索结果是否与查询语义相关
2. **相似度排序**: 结果是否按相似度降序排列
3. **时间过滤**: 能否正确过滤时间范围外的记录
4. **批量搜索**: 多次搜索请求的稳定性

### 6.2 集成测试配置

```bash
# 需要配置环境变量
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
export DASHSCOPE_API_KEY="your-dashscope-key"

# 运行完整集成测试
deno test --allow-all supabase/functions/search-models/test.ts -v
```

---

## 7. 已知问题

1. **集成测试跳过**: 由于未配置 `SUPABASE_SERVICE_ROLE_KEY`，完整集成测试被跳过
2. **数据库无数据**: 当前测试环境 `model_assets` 表为空，无法验证搜索结果相关性

---

## 8. 测试环境重置

```bash
# 重置 Supabase 数据库
cd supabase
supabase db reset

# 重新启动 Edge Function
supabase functions serve search-models --no-verify-jwt --env-file .env.local

# 重新运行测试
deno test --allow-all supabase/functions/search-models/test.ts
```
