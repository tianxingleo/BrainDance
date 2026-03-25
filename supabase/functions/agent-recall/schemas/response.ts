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
});

export type AgentRecallResponse = z.infer<typeof agentRecallResponseSchema>;
