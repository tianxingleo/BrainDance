import { z } from "npm:zod@3.25";

const compareWindowSchema = z.object({
  startTime: z.string(),
  endTime: z.string(),
});

const matchedFrameSchema = z.object({
  imageName: z.string(),
  similarity: z.number().min(0).max(1),
  transformMatrix: z.unknown().nullable(),
  tag: z.string().nullable(),
});

export const compareEvidenceSchema = z.object({
  sceneId: z.string(),
  modelId: z.string(),
  displayName: z.string().nullable(),
  description: z.string().nullable(),
  createdAt: z.string(),
  similarity: z.number().min(0).max(1),
  objects: z.array(z.string()),
  tags: z.array(z.string()),
  matchedFrames: z.array(matchedFrameSchema),
  ply: z.string().nullable(),
  poses: z.string().nullable(),
}).nullable();

const openSceneActionSchema = z.object({
  type: z.literal("open_scene"),
  sceneId: z.string(),
  slot: z.enum(["baseline", "target"]),
  modelId: z.string().nullable().optional(),
  ply: z.string().nullable().optional(),
  poses: z.string().nullable().optional(),
});

const flyToPoseActionSchema = z.object({
  type: z.literal("fly_to_pose"),
  sceneId: z.string(),
  slot: z.enum(["baseline", "target"]),
  imageName: z.string().optional(),
  matrix: z.unknown().nullable().optional(),
});

export const timeCompareActionSchema = z.union([
  openSceneActionSchema,
  flyToPoseActionSchema,
]);

export const timeCompareResponseSchema = z.object({
  success: z.literal(true),
  intent: z.object({
    originalQuery: z.string(),
    parsedSearchText: z.string(),
    compareFocus: z.string().nullable(),
    baseline: compareWindowSchema,
    target: compareWindowSchema,
    reasoning: z.string(),
  }),
  comparison: z.object({
    baseline: compareEvidenceSchema,
    target: compareEvidenceSchema,
    diff: z.object({
      commonObjects: z.array(z.string()),
      addedObjects: z.array(z.string()),
      removedObjects: z.array(z.string()),
      commonTags: z.array(z.string()),
      addedTags: z.array(z.string()),
      removedTags: z.array(z.string()),
      limitations: z.array(z.string()),
    }),
  }),
  answer: z.string(),
  actions: z.array(timeCompareActionSchema),
  toolTrace: z.array(z.object({
    toolName: z.string(),
    args: z.record(z.string(), z.unknown()),
    resultSummary: z.string(),
  })),
});

export type TimeCompareResponse = z.infer<typeof timeCompareResponseSchema>;
