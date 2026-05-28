import { z } from "npm:zod@3.25";

const candidateSceneRefSchema = z.object({
  index: z.number().int().min(1),
  sceneId: z.string(),
  modelId: z.string(),
  description: z.string(),
});

const entitySlotSchema = z.object({
  id: z.string(),
  kind: z.enum(["model", "scene", "location"]),
  label: z.string().max(60),
  mentionedAt: z.number().int().min(0),
  source: z.enum(["result", "user"]),
});

const preferenceMapSchema = z.object({
  regions: z.array(z.string()).max(3).optional(),
  assetTypes: z.array(z.string()).max(3).optional(),
  timeRange: z.string().nullable().optional(),
});

const shortTermMemorySchema = z.object({
  entities: z.array(entitySlotSchema).max(5),
  preferences: preferenceMapSchema,
  turnCount: z.number().int().min(0),
}).nullable().optional();

const sessionStateSchema = z.object({
  lastMode: z.enum([
    "spatial_search",
    "asset_metadata",
    "time_compare",
    "creative",
    "memory_graph",
  ]).optional(),
  lastSelectedModelIds: z.array(z.string()).optional(),
  lastCandidateRefs: z.array(candidateSceneRefSchema).optional(),
  lastOperationPreview: z.object({
    toolName: z.string(),
    affectedCount: z.number().int().min(0),
    modelIds: z.array(z.string()).optional(),
    args: z.record(z.string(), z.unknown()).optional(),
  }).nullable().optional(),
}).nullable().optional();

export const agentRecallRequestSchema = z.object({
  query: z.string().trim().min(1, "搜索语句不能为空").max(
    500,
    "搜索语句过长（最大 500 字符）",
  ),
  selectedModelIds: z.array(z.string()).optional(),
  executionMode: z.enum(["preview", "execute"]).default("execute"),
  currentSceneId: z.string().nullable().optional(),
  currentModelId: z.string().nullable().optional(),
  currentMode: z.enum(["search", "compare", "batch_edit", "collection"])
    .nullable().optional(),
  candidateSceneIds: z.array(z.string()).optional(),
  sessionId: z.string().optional(),
  userId: z.string().optional(),
  conversationSummary: z.string().nullable().optional(),
  shortTermMemory: shortTermMemorySchema,
  sessionState: sessionStateSchema,
});

export type AgentRecallRequest = z.infer<typeof agentRecallRequestSchema>;
