import { z } from "npm:zod@3.25";

export const agentRecallRequestSchema = z.object({
  query: z.string().trim().min(1, "搜索语句不能为空").max(
    500,
    "搜索语句过长（最大 500 字符）",
  ),
  selectedModelIds: z.array(z.string()).optional(),
  executionMode: z.enum(["preview", "execute"]).default("execute"),
  currentSceneId: z.string().nullable().optional(),
  currentModelId: z.string().nullable().optional(),
  candidateSceneIds: z.array(z.string()).optional(),
  sessionId: z.string().optional(),
  conversationSummary: z.string().nullable().optional(),
});

export type AgentRecallRequest = z.infer<typeof agentRecallRequestSchema>;
