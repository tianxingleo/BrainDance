import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import { normalizeDate, safeJsonParse } from "./shared.ts";

function safeGetEnv(name: string): string {
  try {
    return Deno.env.get(name) ?? "";
  } catch {
    return "";
  }
}

function getSupabaseUrl(): string {
  return safeGetEnv("SUPABASE_URL") || "http://127.0.0.1:54321";
}

function getSupabaseKey(): string {
  return safeGetEnv("SUPABASE_SERVICE_ROLE_KEY");
}

function getDashscopeApiKey(): string {
  return safeGetEnv("DASHSCOPE_API_KEY");
}

Deno.test("safeJsonParse - null 输入", () => {
  const result = safeJsonParse(null);
  assertEquals(Object.keys(result).length, 0);
});

Deno.test("safeJsonParse - 无效 JSON 输入", () => {
  const result = safeJsonParse("not json");
  assertEquals(Object.keys(result).length, 0);
});

Deno.test("normalizeDate - 有效日期格式", () => {
  const result = normalizeDate("2026-01-20T10:30:00Z");
  assertEquals(result, "2026-01-20T10:30:00Z");
});

Deno.test("normalizeDate - 无效日期格式", () => {
  const result = normalizeDate("2026/01/20");
  assertEquals(result, null);
});

Deno.test({
  name: "API 测试 - 缺少 query 参数",
  async fn() {
    const supabaseKey = getSupabaseKey();
    if (!supabaseKey) return;

    const resp = await fetch(`${getSupabaseUrl()}/functions/v1/search-models`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${supabaseKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({}),
    });

    assertEquals(resp.status, 400);
  },
  sanitizeResources: false,
  sanitizeOps: false,
});

Deno.test({
  name: "集成测试 - 完整搜索流程",
  async fn() {
    const dashscopeApiKey = getDashscopeApiKey();
    const supabaseKey = getSupabaseKey();
    if (!dashscopeApiKey || !supabaseKey) {
      return;
    }

    const resp = await fetch(`${getSupabaseUrl()}/functions/v1/search-models`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${supabaseKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: "测试搜索关键词" }),
    });

    assertEquals(resp.status < 500, true);
    const data = await resp.json();
    assertEquals(typeof data.success, "boolean");
  },
  sanitizeResources: false,
  sanitizeOps: false,
});
