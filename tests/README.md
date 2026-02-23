# BrainDance Deno API 测试工具

本目录包含用于测试 BrainDance Deno Edge Function 的工具集。

## 📁 文件说明

### 1. HTML 测试页面（推荐）
**文件**: [search-test.html](./search-test.html)

美观的可视化测试界面，无需任何依赖。

### 2. Python 自动化测试脚本
**文件**: [test_search_api.py](./test_search_api.py)

自动化测试套件，包含 6 个测试用例。

支持从环境变量读取配置（优先级从高到低）：

1. `SEARCH_API_URL` - 完整的 API URL（例如：`https://api.ratherhard.com/functions/v1/search-models`）
2. `SUPABASE_URL` - Supabase 项目 URL（自动构建 API URL，例如：`https://api.ratherhard.com`）
3. 默认本地地址：`http://127.0.0.1:54321/functions/v1/search-models`

**自动加载 .env 文件**：
- `tests/.env.local`（测试专用配置）
- `ai_engine/3dgs/.env`（共享配置）
- `tests/.env`（通用配置）

## 🚀 快速开始

### 步骤 1: 启动 Edge Function

```bash
cd /home/ltx/projects/BrainDance/supabase
supabase functions serve search-models --no-verify-jwt --env-file functions/search-models/.env.local
```

预期输出:
```
Started Edge Functions runtime on http://127.0.0.1:54321
```

### 步骤 2: 运行测试

#### 方式 A: 使用 HTML 页面（推荐）

直接在浏览器中打开 `search-test.html` 文件：

```bash
# 使用默认浏览器
firefox /home/ltx/projects/BrainDance/tests/search-test.html

# 或启动本地服务器
python3 -m http.server 8080 --directory /home/ltx/projects/BrainDance/tests
# 然后访问 http://localhost:8080/search-test.html
```

#### 方式 B: 运行 Python 测试

```bash
# 确保已安装 requests
pip install requests

# 运行测试
cd /home/ltx/projects/BrainDance
python3 tests/test_search_api.py
```

## 📊 Python 测试说明

### 测试用例

| 测试 | 说明 |
|------|------|
| 1. 基础搜索 | 验证基本的搜索功能 |
| 2. 时间范围搜索 | 验证时间范围提取和过滤 |
| 3. 高阈值搜索 | 验证阈值参数功能 |
| 4. 空查询 | 验证参数验证（应失败） |
| 5. 超长查询 | 验证长度限制（应失败） |
| 6. 响应格式 | 验证响应数据结构 |

### 使用示例

```bash
# 方式 1: 使用默认本地 URL
python3 tests/test_search_api.py

# 方式 2: 使用现有配置（自动加载 ai_engine/3dgs/.env）
pip install python-dotenv  # 先安装 python-dotenv
python3 tests/test_search_api.py  # 自动读取 ai_engine/3dgs/.env 中的 SUPABASE_URL

# 方式 3: 使用环境变量
export SEARCH_API_URL=https://api.ratherhard.com/functions/v1/search-models
python3 tests/test_search_api.py

# 方式 4: 使用自定义 .env.local 文件
cp tests/.env.example tests/.env.local
# 编辑 tests/.env.local:
# SEARCH_API_URL=https://your-project.supabase.co/functions/v1/search-models
python3 tests/test_search_api.py  # 自动加载 tests/.env.local

# 方式 5: 命令行参数（最高优先级）
python3 tests/test_search_api.py https://custom-url.com/functions/v1/search-models
```

### 配置优先级

测试脚本按以下优先级读取 API URL：

1. **命令行参数** - 直接传入 URL
2. `SEARCH_API_URL` 环境变量
3. `SUPABASE_URL` 环境变量（自动构建完整 URL）
4. 默认值：`http://127.0.0.1:54321/functions/v1/search-models`

### 预期输出

```
============================================================
BrainDance 搜索 API 测试套件
============================================================
测试时间: 2026-01-23 15:30:00
API 端点: http://127.0.0.1:54321/functions/v1/search-models

[测试 1] 基础搜索测试
✓ 通过 - 找到 3 个结果

[测试 2] 时间范围搜索测试
✓ 通过 - 时间范围: 2026-01-13T00:00:00Z 至 2026-01-19T23:59:59Z

...

============================================================
测试报告
============================================================

总测试数: 6
通过: 6 ✓
失败: 0 ✗
通过率: 100.0%
平均响应时间: 0.65s

🎉 所有测试通过!
```

## 🔧 API 信息

### 配置说明

测试脚本支持多种配置方式，优先级从高到低：

1. **命令行参数** - 直接传入 URL
2. **环境变量** - 设置 `SEARCH_API_URL`
3. **默认值** - `http://127.0.0.1:54321/functions/v1/search-models`

**注意**: Edge Function 使用的 `DASHSCOPE_API_KEY` 在 `supabase/functions/search-models/.env.local` 中配置，测试脚本不需要这个 key。

### 端点

- **URL**: `http://127.0.0.1:54321/functions/v1/search-models`
- **方法**: POST
- **Content-Type**: application/json

### 请求格式

```json
{
  "query": "搜索关键词",
  "threshold": 0.5
}
```

### 响应格式

```json
{
  "success": true,
  "intent": {
    "original_query": "原始查询",
    "parsed_search_text": "解析后的搜索词",
    "filter_start": "时间范围起点",
    "filter_end": "时间范围终点"
  },
  "threshold": 0.5,
  "results": [
    {
      "id": "uuid",
      "scene_id": "场景ID",
      "description": "场景描述",
      "similarity": 0.85,
      "created_at": "2026-01-15T14:30:22Z",
      "ply_path": "模型文件路径"
    }
  ]
}
```

## 🐛 故障排查

### 问题 1: 连接被拒绝

**症状**: `Failed to connect to 127.0.0.1:54321`

**解决方案**:
```bash
# 检查 Edge Function 是否运行
ps aux | grep deno

# 重新启动
cd /home/ltx/projects/BrainDance/supabase
supabase functions serve search-models --no-verify-jwt --env-file functions/search-models/.env.local
```

### 问题 2: CORS 错误（HTML 页面）

**症状**: 浏览器控制台显示 CORS 错误

**解决方案**:
使用本地 HTTP 服务器而不是直接打开文件：
```bash
python3 -m http.server 8080 --directory /home/ltx/projects/BrainDance/tests
```

### 问题 3: API Key 错误

**症状**: `未配置 DASHSCOPE_API_KEY`

**解决方案**:
检查 `.env.local` 文件：
```bash
cat /home/ltx/projects/BrainDance/supabase/functions/search-models/.env.local
```

确保包含：
```
DASHSCOPE_API_KEY=sk-xxxxx
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

### 问题 4: 没有搜索结果

**症状**: 所有测试通过但结果为空

**原因**: 数据库中没有向量数据

**解决**: 需要先运行 3DGS 生成流程创建一些测试数据。

## 📝 依赖安装

### Python 依赖

```bash
# 必需依赖
pip install requests

# 可选依赖（用于自动加载 .env 文件）
pip install python-dotenv
```

**注意**: 如果不安装 `python-dotenv`，测试脚本仍然可以正常运行，只是不会自动加载 .env 文件。

## 📚 相关文档

- [Edge Function 源码](../supabase/functions/search-models/index.ts)
- [Edge Function 测试](../supabase/functions/search-models/test.ts)
- [数据库 Schema](../supabase/migrations/)
