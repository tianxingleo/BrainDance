import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import { agentRecallRequestSchema } from "./schemas/request.ts";

Deno.test("agentRecallRequestSchema 可以正确解析包含 executionMode 和 selectedModelIds 的请求", () => {
  const result = agentRecallRequestSchema.safeParse({
    query: "把这三个模型统加上宿舍标签",
    selectedModelIds: ["uuid1", "uuid2"],
    executionMode: "preview",
  });

  assertEquals(result.success, true);
  if (result.success) {
    assertEquals(result.data.query, "把这三个模型统加上宿舍标签");
    assertEquals(result.data.selectedModelIds, ["uuid1", "uuid2"]);
    assertEquals(result.data.executionMode, "preview");
  }
});

Deno.test("agentRecallRequestSchema 在没有 executionMode 时默认使用 execute", () => {
  const result = agentRecallRequestSchema.safeParse({
    query: "找一下黑色的耳机",
  });

  assertEquals(result.success, true);
  if (result.success) {
    assertEquals(result.data.executionMode, "execute");
    assertEquals(result.data.selectedModelIds, undefined);
  }
});

Deno.test("agentRecallRequestSchema 应拒绝过长的 query", () => {
  const result = agentRecallRequestSchema.safeParse({
    query: "a".repeat(501),
  });

  assertEquals(result.success, false);
});
