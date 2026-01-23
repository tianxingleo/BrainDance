# TODO: Auth 权限迁移计划

> 将 Edge Function 从 `Service Role Key` 切换到 `Anon Key + 用户认证`

## 背景

当前 Edge Function 使用 `SUPABASE_SERVICE_ROLE_KEY` 运行，拥有绕过 RLS 的完整数据库权限。这适合本地开发和快速迭代，但在生产环境中存在安全隐患：

- **风险**: Service Role Key 拥有管理员权限，任何能调用 API 的用户都能访问/修改所有数据
- **合规**: 不符合最小权限原则

## 目标

切换到 `Anon Key + 用户认证`，实现：
- 用户只能搜索自己有权限访问的资产
- 遵循 Supabase RLS (Row Level Security) 策略
- 最小化权限暴露

## 前置条件

在执行迁移前，确保以下条件已满足：

1. ✅ Flutter 应用已实现用户登录/注册功能
2. ✅ `model_assets` 表已配置 RLS 策略
3. ✅ 用户资产关联通过 `user_id` 字段实现
4. ✅ Edge Function 已通过本地测试验证

## 迁移步骤

### 阶段 1: 数据库 RLS 配置

#### 1.1 启用 RLS

```sql
-- 启用 RLS
ALTER TABLE model_assets ENABLE ROW LEVEL SECURITY;

-- 创建策略：用户只能查看自己的资产
CREATE POLICY "Users can view own assets"
ON model_assets
FOR SELECT
USING (auth.uid()::text = user_id);
```

#### 1.2 验证 RLS

```sql
-- 检查策略
SELECT * FROM pg_policies WHERE tablename = 'model_assets';

-- 测试策略
SET ROLE authenticated;
SELECT * FROM model_assets; -- 应只返回当前用户的资产
```

### 阶段 2: Edge Function 修改

#### 2.1 切换到 Anon Key

**修改前**:
```typescript
const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '';
```

**修改后**:
```typescript
// 使用用户请求中的 Authorization Header
const authHeader = req.headers.get('Authorization');
if (!authHeader) {
  return errorResponse("未提供认证令牌", 401);
}

const supabase = createClient(supabaseUrl, authHeader);
```

#### 2.2 添加用户上下文

```typescript
// 获取当前用户
const { data: { user }, error: authError } = await supabase.auth.getUser();

if (authError || !user) {
  return errorResponse("认证失败", 401);
}

console.log(`[Search] User ${user.id} is searching...`);
```

#### 2.3 修改 RPC 调用

**当前实现** (Service Role):
```typescript
const { data: results, error } = await supabase.rpc('match_model_assets', {
  query_embedding: vector,
  match_threshold: 0.7,
  match_count: 10
});
```

**迁移后** (Anon Key + RLS):
```typescript
// RPC 函数也需要支持用户过滤
// 建议：修改 match_model_assets 添加 user_id 参数

const { data: results, error } = await supabase.rpc('match_model_assets', {
  query_embedding: vector,
  match_threshold: 0.7,
  match_count: 10,
  filter_user_id: user.id  // 添加用户过滤
});
```

### 阶段 3: 创建 RPC 升级迁移

```sql
-- 新的 match_model_assets 函数 (支持用户过滤)
CREATE OR REPLACE FUNCTION public.match_model_assets(
    query_embedding public.vector,
    match_threshold double precision,
    match_count integer,
    filter_start timestamp with time zone DEFAULT NULL,
    filter_end timestamp with time zone DEFAULT NULL,
    filter_user_id text DEFAULT NULL  -- 新增参数
)
RETURNS TABLE(id uuid, scene_id text, description text, ply_path text, created_at timestamp with time zone, similarity double precision)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    m.id, m.scene_id, m.description, m.ply_path, m.created_at,
    1 - (m.embedding <=> query_embedding) AS similarity
  FROM model_assets m
  WHERE
    (filter_user_id IS NULL OR m.user_id = filter_user_id)
    AND (filter_start IS NULL OR m.created_at >= filter_start)
    AND (filter_end IS NULL OR m.created_at <= filter_end)
    AND (1 - (m.embedding <=> query_embedding)) > match_threshold
  ORDER BY m.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

### 阶段 4: 测试验证

#### 4.1 测试场景

| 场景 | 预期结果 |
|------|----------|
| 未登录用户调用 | 返回 401 |
| 普通用户调用 | 只返回自己的资产 |
| 跨用户访问 | 返回空结果 (RLS 阻止) |
| Admin Service Role 调用 | 仍可访问全部数据 |

#### 4.2 测试命令

```bash
# 未登录
curl -i -X POST http://127.0.0.1:54321/functions/v1/search-models \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# 使用用户 Token
curl -i -X POST http://127.0.0.1:54321/functions/v1/search-models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -d '{"query": "test"}'
```

## 回滚计划

如果迁移出现问题，快速回滚：

```bash
# 1. 恢复使用 Service Role
# 在 Edge Function 中切换回 SUPABASE_SERVICE_ROLE_KEY

# 2. 或降级 API 版本
supabase functions restore search-models --version previous-version
```

## 检查清单 (Checklist)

- [ ] RLS 策略已创建并测试
- [ ] Edge Function 已修改支持用户认证
- [ ] RPC 函数已升级支持 filter_user_id
- [ ] 所有测试场景通过
- [ ] 回滚方案已验证可用
- [ ] 文档已更新

## 时间线建议

| 阶段 | 时间 | 负责人 |
|------|------|--------|
| 数据库 RLS 配置 | 1 天 | 后端开发者 |
| Edge Function 修改 | 1 天 | 后端开发者 |
| 测试与修复 | 2 天 | QA + 后端 |
| 上线与监控 | 1 天 | DevOps |

## 风险与缓解

| 风险 | 严重程度 | 缓解措施 |
|------|----------|----------|
| RLS 配置错误 | 高 | 先在测试环境验证 |
| 性能下降 | 中 | 添加索引优化查询 |
| Token 过期 | 低 | 实现 Token 自动刷新 |

## 参考文档

- [Supabase RLS 官方文档](https://supabase.com/docs/guides/auth/row-level-security)
- [Edge Function 认证](https://supabase.com/docs/guides/functions/auth)
- [现有实现参考](../代办/Supabase Edge Function (Deno) _整理版.md)

---

## 状态

- [ ] 未开始
- [ ] 进行中
- [ ] 已完成

**创建日期**: 2026-01-20  
**预计完成时间**: 待定
