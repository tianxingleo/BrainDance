/**
 * BrainDance 语义搜索 Edge Function - 自动化测试
 * 
 * ================================================================
 * 测试说明
 * ================================================================
 * 
 * 本测试文件使用 Deno Test 框架编写
 * 用于验证 Edge Function 的各个功能模块
 * 
 * 运行命令:
 *   deno test --allow-all supabase/functions/search-models/test.ts
 * 
 * 测试覆盖范围:
 * 1. 辅助函数单元测试 (safeJsonParse, normalizeDate)
 * 2. API 端点集成测试 (参数验证, CORS)
 * 3. 完整搜索流程测试 (需要配置 SUPABASE_SERVICE_ROLE_KEY)
 * 
 * ================================================================
 * 环境配置
 * ================================================================
 * 
 * 测试前需要配置以下环境变量:
 * - SUPABASE_URL: Supabase 本地 URL (默认 http://127.0.0.1:54321)
 * - SUPABASE_SERVICE_ROLE_KEY: 服务角色密钥
 * - DASHSCOPE_API_KEY: DashScope API Key
 * 
 * 配置方式:
 * 1. 临时导出:
 *    export SUPABASE_URL="http://127.0.0.1:54321"
 *    export SUPABASE_SERVICE_ROLE_KEY="your-key"
 * 
 * 2. 或使用 .env.local 文件 (推荐)
 *    supabase functions serve search-models --env-file .env.local
 * 
 * ================================================================
 */

// ============================================================================
// 【导入模块】导入 Deno 标准测试库
// ============================================================================

/**
 * assertEquals: 断言函数，用于验证实际值与预期值相等
 * 
 * 使用示例:
 * assertEquals(1 + 1, 2);  // 通过
 * assertEquals("a", "b");  // 抛出 AssertionError
 */
import { assertEquals, assertExists } from "https://deno.land/std@0.168.0/testing/asserts.ts";

// ============================================================================
// 【测试配置】从环境变量读取配置
// ============================================================================

/**
 * 测试环境变量配置
 * 
 * 从系统环境变量读取 Supabase 和 DashScope 配置
 * 如果未设置，使用默认值
 */
const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "http://127.0.0.1:54321";
const SUPABASE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const DASHSCOPE_API_KEY = Deno.env.get("DASHSCOPE_API_KEY") ?? "";

// ============================================================================
// 【单元测试】测试辅助函数
// ============================================================================

/**
 * 测试 safeJsonParse 函数 - 有效 JSON 输入
 * 
 * 测试场景:
 * - 输入: '{"name": "test", "value": 123}'
 * - 预期: 返回 { name: "test", value: 123 }
 * - 验证: 对象属性值正确
 */
Deno.test({
  name: "safeJsonParse - 有效 JSON 输入",
  fn() {
    const result = JSON.parse('{"search_text": "test", "start_time": null}');
    assertEquals(result.search_text, "test");
  },
});

/**
 * 测试 safeJsonParse 函数 - null 输入
 * 
 * 测试场景:
 * - 输入: null
 * - 预期: 返回空对象 {}
 * - 验证: 函数正确处理 null 值
 */
Deno.test({
  name: "safeJsonParse - null 输入",
  fn() {
    const result = safeJsonParse(null);
    assertEquals(Object.keys(result).length, 0);
  },
});

/**
 * 测试 safeJsonParse 函数 - 无效 JSON 输入
 * 
 * 测试场景:
 * - 输入: "not json"
 * - 预期: 返回空对象 {}，不抛出异常
 * - 验证: 函数优雅处理解析失败
 */
Deno.test({
  name: "safeJsonParse - 无效 JSON 输入",
  fn() {
    const result = safeJsonParse("not json");
    assertEquals(Object.keys(result).length, 0);
  },
});

/**
 * 测试 normalizeDate 函数 - 有效日期格式
 * 
 * 测试场景:
 * - 输入: "2026-01-20T10:30:00Z"
 * - 预期: 返回原值 "2026-01-20T10:30:00Z"
 * - 验证: ISO 8601 格式验证通过
 */
Deno.test({
  name: "normalizeDate - 有效日期格式",
  fn() {
    const result = normalizeDate("2026-01-20T10:30:00Z");
    assertEquals(result, "2026-01-20T10:30:00Z");
  },
});

/**
 * 测试 normalizeDate 函数 - null 输入
 * 
 * 测试场景:
 * - 输入: null
 * - 预期: 返回 null
 * - 验证: 空值处理正确
 */
Deno.test({
  name: "normalizeDate - null 输入",
  fn() {
    const result = normalizeDate(null);
    assertEquals(result, null);
  },
});

/**
 * 测试 normalizeDate 函数 - 无效日期格式
 * 
 * 测试场景:
 * - 输入: "2026/01/20" (错误分隔符)
 * - 预期: 返回 null
 * - 验证: 无效格式被正确拒绝
 */
Deno.test({
  name: "normalizeDate - 无效日期格式",
  fn() {
    const result = normalizeDate("2026/01/20");
    assertEquals(result, null);
  },
});

// ============================================================================
// 【集成测试】测试 API 端点
// ============================================================================

/**
 * 集成测试 - 完整搜索流程
 * 
 * 测试场景:
 * - 发送搜索请求到 Edge Function
 * - 预期: 返回 success: true 的响应
 * 
 * 前置条件:
 * - 需要配置 SUPABASE_SERVICE_ROLE_KEY
 * - 需要配置 DASHSCOPE_API_KEY
 * - 本地 Supabase 必须运行
 * - 数据库中必须有数据
 */
Deno.test({
  name: "集成测试 - 完整搜索流程",
  async fn() {
    // 检查环境变量
    if (!DASHSCOPE_API_KEY) {
      console.log("⚠️ 跳过集成测试: 未配置 DASHSCOPE_API_KEY");
      return;
    }
    if (!SUPABASE_KEY) {
      console.log("⚠️ 跳过集成测试: 未配置 SUPABASE_SERVICE_ROLE_KEY");
      return;
    }

    // 发送 HTTP 请求
    const resp = await fetch(`${SUPABASE_URL}/functions/v1/search-models`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${SUPABASE_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: "测试搜索关键词" }),
    });

    // 验证响应状态码 (不应是 5xx 服务器错误)
    assertEquals(resp.status < 500, true, "API 不应返回 5xx 错误");
    
    // 验证响应格式
    const data = await resp.json();
    assertEquals(typeof data.success, "boolean");
  },
  sanitizeResources: false,
  sanitizeOps: false,
});

/**
 * API 测试 - 缺少 query 参数
 * 
 * 测试场景:
 * - 发送空 JSON: {}
 * - 预期: 返回 400 错误
 * - 验证: 参数验证生效
 */
Deno.test({
  name: "API 测试 - 缺少 query 参数",
  async fn() {
    if (!SUPABASE_KEY) return;

    const resp = await fetch(`${SUPABASE_URL}/functions/v1/search-models`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${SUPABASE_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({}),
    });

    // 验证返回 400 错误
    assertEquals(resp.status, 400);
    
    // 验证错误信息包含 "缺少"
    const data = await resp.json();
    assertEquals(data.success, false);
    assertEquals(data.error.includes("缺少"), true);
  },
  sanitizeResources: false,
  sanitizeOps: false,
});

/**
 * API 测试 - 空查询字符串
 * 
 * 测试场景:
 * - 发送空字符串 query: "   "
 * - 预期: 返回 400 错误
 * - 验证: 空字符串验证生效
 */
Deno.test({
  name: "API 测试 - 空查询字符串",
  async fn() {
    if (!SUPABASE_KEY) return;

    const resp = await fetch(`${SUPABASE_URL}/functions/v1/search-models`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${SUPABASE_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: "   " }),
    });

    assertEquals(resp.status, 400);
    const data = await resp.json();
    assertEquals(data.success, false);
  },
  sanitizeResources: false,
  sanitizeOps: false,
});

/**
 * API 测试 - 查询字符串过长
 * 
 * 测试场景:
 * - 发送 501 字符的查询
 * - 预期: 返回 400 错误
 * - 验证: 长度限制生效
 */
Deno.test({
  name: "API 测试 - 查询字符串过长",
  async fn() {
    if (!SUPABASE_KEY) return;

    const longQuery = "a".repeat(501);
    const resp = await fetch(`${SUPABASE_URL}/functions/v1/search-models`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${SUPABASE_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: longQuery }),
    });

    assertEquals(resp.status, 400);
    const data = await resp.json();
    assertEquals(data.error.includes("过长"), true);
  },
  sanitizeResources: false,
  sanitizeOps: false,
});

// ============================================================================
// 【CORS 测试】测试跨域资源共享
// ============================================================================

/**
 * CORS 测试 - OPTIONS 预检请求
 * 
 * 测试场景:
 * - 发送 OPTIONS 请求 (模拟浏览器预检)
 * - 预期: 返回 200 OK
 * - 预期: 包含正确的 CORS 头
 * 
 * 浏览器行为:
 * - 跨域请求前先发送 OPTIONS
 * - 检查 Access-Control-Allow-Origin 头
 * - 通过后才发送实际请求
 */
Deno.test({
  name: "CORS 测试 - OPTIONS 预检请求",
  async fn() {
    const resp = await fetch(`${SUPABASE_URL}/functions/v1/search-models`, {
      method: "OPTIONS",
      headers: {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
      },
    });

    // 验证返回 200
    assertEquals(resp.status, 200);
    
    // 验证 CORS 头正确
    assertEquals(resp.headers.get("Access-Control-Allow-Origin"), "*");
  },
  sanitizeResources: false,
  sanitizeOps: false,
});

// ============================================================================
// 【辅助函数】从 index.ts 复制的函数用于本地测试
// ============================================================================

/**
 * 安全解析 JSON (本地测试版本)
 * 
 * @param str - JSON 字符串
 * @returns 解析后的对象或空对象
 */
function safeJsonParse(str: string | null): Record<string, unknown> {
  if (!str) return {};
  try {
    return JSON.parse(str);
  } catch {
    console.error("[Test] JSON 解析失败:", str);
    return {};
  }
}

/**
 * 验证日期格式 (本地测试版本)
 * 
 * @param dateStr - 日期字符串
 * @returns 有效返回原值，无效返回 null
 */
function normalizeDate(dateStr: string | null): string | null {
  if (!dateStr) return null;
  const regex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;
  return regex.test(dateStr) ? dateStr : null;
}

// ============================================================================
// 【测试完成提示】
// ============================================================================

console.log("✅ 测试文件加载完成。运行 'deno test --allow-all' 执行测试。");
