import { z } from "npm:zod@3.25";

export const matchedFrameSchema = z.object({
  imageName: z.string(),
  similarity: z.number().min(0).max(1),
  transformMatrix: z.unknown().nullable(),
});

export const evidenceSchema = z.object({
  sceneId: z.string(),
  similarity: z.number().min(0).max(1),
  matchedFrames: z.array(matchedFrameSchema),
}).nullable();

export const openSceneActionSchema = z.object({
  type: z.literal("open_scene"),
  sceneId: z.string(),
  modelId: z.string().nullable().optional(),
  ply: z.string().nullable().optional(),
  poses: z.string().nullable().optional(),
});

export const flyToPoseActionSchema = z.object({
  type: z.literal("fly_to_pose"),
  sceneId: z.string(),
  imageName: z.string().optional(),
  matrix: z.unknown().nullable().optional(),
});

export const agentRecallActionSchema = z.union([
  openSceneActionSchema,
  flyToPoseActionSchema,
]);

export const agentRecallResponseSchema = z.object({
  answer: z.string(),
  evidence: evidenceSchema,
  actions: z.array(agentRecallActionSchema),
  top_candidates: z.array(z.unknown()).optional(),
  selected_candidate_reason: z.string().optional(),
});

export const agentRecallPlanEventSchema = z.object({
  event: z.literal("plan"),
  data: z.object({
    title: z.string(),
    steps: z.array(z.string()),
  }),
});

export const agentRecallThinkingEventSchema = z.object({
  event: z.literal("thinking"),
  data: z.object({
    content: z.string(),
  }),
});

export const agentRecallToolCallEventSchema = z.object({
  event: z.literal("tool_call"),
  data: z.object({
    name: z.string(),
    args: z.record(z.string(), z.unknown()),
  }),
});

export const agentRecallToolResultEventSchema = z.object({
  event: z.literal("tool_result"),
  data: z.object({
    name: z.string(),
    status: z.enum(["success", "empty", "error"]),
    result: z.unknown(),
  }),
});

export const agentRecallMessageEventSchema = z.object({
  event: z.literal("message"),
  data: z.object({
    delta: z.string(),
  }),
});

export const agentRecallDoneEventSchema = z.object({
  event: z.literal("done"),
  data: agentRecallResponseSchema,
});

export const agentRecallErrorEventSchema = z.object({
  event: z.literal("error"),
  data: z.object({
    message: z.string(),
  }),
});

export const agentRecallStreamEventSchema = z.discriminatedUnion("event", [
  agentRecallPlanEventSchema,
  agentRecallThinkingEventSchema,
  agentRecallToolCallEventSchema,
  agentRecallToolResultEventSchema,
  agentRecallMessageEventSchema,
  agentRecallDoneEventSchema,
  agentRecallErrorEventSchema,
]);

export type AgentRecallResponse = z.infer<typeof agentRecallResponseSchema>;
export type AgentRecallStreamEvent = z.infer<
  typeof agentRecallStreamEventSchema
>;
