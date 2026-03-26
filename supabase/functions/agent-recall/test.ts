import { assertEquals } from "https://deno.land/std@0.168.0/testing/asserts.ts";
import { agentRecallRequestSchema } from "./schemas/request.ts";

Deno.test("agentRecallRequestSchema 可以正确解析包含上下文字段的请求", () => {
  const result = agentRecallRequestSchema.safeParse({
    query: "把这三个模型统加上宿舍标签",
    selectedModelIds: ["uuid1", "uuid2"],
    executionMode: "preview",
    currentSceneId: "scene-1",
    currentModelId: "model-1",
    currentMode: "batch_edit",
    conversationSummary: "上一轮已经确认这是宿舍相关模型。",
    sessionState: {
      lastMode: "spatial_search",
      lastSelectedModelIds: ["uuid1", "uuid2"],
      lastCandidateRefs: [
        {
          index: 1,
          sceneId: "scene-a",
          modelId: "model-a",
          description: "宿舍书桌",
        },
      ],
      lastOperationPreview: {
        toolName: "batch_patch_model_metadata",
        affectedCount: 2,
        modelIds: ["uuid1", "uuid2"],
        args: {
          modelIds: ["uuid1", "uuid2"],
          patch: {
            displayNameTemplate: "宿舍-归档版",
            tagsAdd: [],
            tagsRemove: [],
          },
          dryRun: true,
        },
      },
    },
  });

  assertEquals(result.success, true);
  if (result.success) {
    assertEquals(result.data.query, "把这三个模型统加上宿舍标签");
    assertEquals(result.data.selectedModelIds, ["uuid1", "uuid2"]);
    assertEquals(result.data.executionMode, "preview");
    assertEquals(result.data.currentMode, "batch_edit");
    assertEquals(result.data.sessionState?.lastMode, "spatial_search");
    assertEquals(
      result.data.sessionState?.lastOperationPreview?.modelIds,
      ["uuid1", "uuid2"],
    );
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
