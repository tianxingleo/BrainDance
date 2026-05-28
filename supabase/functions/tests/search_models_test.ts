/**
 * Smoke tests: search-models Edge Function 的输入输出 schema 验证。
 *
 * 纯本地测试，不依赖 Supabase / DashScope 连接。
 * 运行方式: deno test supabase/functions/tests/ --allow-read
 */

import {
  assertEquals,
  assertNotEquals,
} from "https://deno.land/std@0.168.0/testing/asserts.ts";

import {
  normalizeDate,
  safeJsonParse,
  type SearchModelsResponse,
  type SearchResultRow,
} from "../search-models/shared.ts";

// ─── safeJsonParse ─────────────────────────────────────────────────

Deno.test("[schema] safeJsonParse - 有效 JSON 对象", () => {
  const result = safeJsonParse('{"a": 1, "b": "two"}');
  assertEquals(result.a, 1);
  assertEquals(result.b, "two");
});

Deno.test("[schema] safeJsonParse - null 输入返回空对象", () => {
  const result = safeJsonParse(null);
  assertEquals(Object.keys(result).length, 0);
});

Deno.test("[schema] safeJsonParse - 空字符串返回空对象", () => {
  const result = safeJsonParse("");
  assertEquals(Object.keys(result).length, 0);
});

Deno.test("[schema] safeJsonParse - 无效 JSON 返回空对象", () => {
  const result = safeJsonParse("{broken");
  assertEquals(Object.keys(result).length, 0);
});

Deno.test("[schema] safeJsonParse - 嵌套结构", () => {
  const result = safeJsonParse('{"search_text": "桌子", "start_time": null}');
  assertEquals(result.search_text, "桌子");
  assertEquals(result.start_time, null);
});

// ─── normalizeDate ─────────────────────────────────────────────────

Deno.test("[schema] normalizeDate - 有效 ISO8601 UTC 格式", () => {
  assertEquals(
    normalizeDate("2026-03-15T14:30:00Z"),
    "2026-03-15T14:30:00Z",
  );
});

Deno.test("[schema] normalizeDate - null 输入返回 null", () => {
  assertEquals(normalizeDate(null), null);
});

Deno.test("[schema] normalizeDate - 空字符串返回 null", () => {
  assertEquals(normalizeDate(""), null);
});

Deno.test("[schema] normalizeDate - 非 UTC 时区格式拒绝", () => {
  assertEquals(normalizeDate("2026-03-15T14:30:00+08:00"), null);
});

Deno.test("[schema] normalizeDate - 仅日期无时间拒绝", () => {
  assertEquals(normalizeDate("2026-03-15"), null);
});

Deno.test("[schema] normalizeDate - 完全无关字符串拒绝", () => {
  assertEquals(normalizeDate("not a date"), null);
});

// ─── SearchModelsResponse / SearchResultRow 类型构造 ───────────────

Deno.test("[schema] SearchModelsResponse - 最小合法结构", () => {
  const resp: SearchModelsResponse = {
    success: true,
    intent: {
      original_query: "杯子",
      parsed_search_text: "杯子",
      filter_start: null,
      filter_end: null,
    },
    threshold: 0.5,
    results: [],
  };
  assertEquals(resp.success, true);
  assertEquals(resp.results.length, 0);
  assertNotEquals(resp.intent, null);
});

Deno.test("[schema] SearchModelsResponse - 含时间过滤和结果行", () => {
  const resp: SearchModelsResponse = {
    success: true,
    intent: {
      original_query: "红色杯子",
      parsed_search_text: "红色杯子",
      filter_start: "2026-01-01T00:00:00Z",
      filter_end: "2026-01-31T23:59:59Z",
    },
    threshold: 0.7,
    results: [
      { id: "550e8400-e29b-41d4-a716-446655440000", score: 0.85, scene_id: "scene_a" },
    ],
  };
  assertEquals(resp.intent.filter_start, "2026-01-01T00:00:00Z");
  assertEquals(resp.results.length, 1);
  assertEquals(resp.results[0]["score"], 0.85);
});

Deno.test("[schema] SearchResultRow - 兼容任意额外字段", () => {
  const row: SearchResultRow = {
    id: "abc",
    scene_id: "scene_x",
    display_name: "客厅扫描",
    custom_field: 42,
  };
  assertEquals(row["custom_field"], 42);
  assertEquals(row["display_name"], "客厅扫描");
});

Deno.test("[schema] SearchModelsResponse - 阈值边界值", () => {
  // 阈值在 [0, 1] 范围内应有效
  for (const threshold of [0, 0.5, 1]) {
    const resp: SearchModelsResponse = {
      success: true,
      intent: {
        original_query: "test",
        parsed_search_text: "test",
        filter_start: null,
        filter_end: null,
      },
      threshold,
      results: [],
    };
    assertEquals(resp.threshold, threshold);
  }
});
