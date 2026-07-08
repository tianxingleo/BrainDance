import { z } from "npm:zod@3.25";

export const timeCompareRequestSchema = z.object({
  query: z.string().trim().min(1, "搜索语句不能为空").max(
    500,
    "搜索语句过长（最大 500 字符）",
  ),
  threshold: z.number().min(0).max(1).optional(),
});

export type TimeCompareRequest = z.infer<typeof timeCompareRequestSchema>;
