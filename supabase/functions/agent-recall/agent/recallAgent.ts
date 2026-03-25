import { searchSpace } from "../tools/searchSpace.ts";
import { buildEvidenceFromSpatialResult } from "../tools/getSceneAsset.ts";
import { buildRecallActionsFromSearchResult } from "../tools/buildViewAction.ts";
import {
  type AgentRecallResponse,
  agentRecallResponseSchema,
} from "../schemas/response.ts";

function buildAnswerFromSearchResult(
  result: Awaited<ReturnType<typeof searchSpace>>,
): string {
  const topResult = result.results[0];
  if (!topResult) {
    return "当前没有找到可信的空间检索结果。";
  }

  const description = typeof topResult.description === "string"
    ? topResult.description
    : result.intent.parsed_search_text;
  
  // 截断 description 以防止上下文溢出
  const truncatedDescription = description && description.length > 2000 
    ? description.substring(0, 2000) + "...[内容过长已截断]" 
    : description;

  const displayName = typeof topResult.display_name === "string"
    ? topResult.display_name.trim()
    : "";
  const sceneId = displayName.length > 0
    ? displayName
    : typeof topResult.scene_id === "string"
    ? topResult.scene_id
    : "未知场景";
  const similarity = Math.round(Number(topResult.similarity ?? 0) * 100);
  return `已找到最相关的空间记忆，命中场景 ${sceneId}，相关描述为“${truncatedDescription}”，相似度约 ${similarity}%。`;
}

export async function runRecallAgent(
  query: string,
  signal?: AbortSignal,
): Promise<AgentRecallResponse> {
  // 在耗时操作前检查中断信号
  if (signal?.aborted) {
    throw new Error("任务已取消");
  }

  const searchResult = await searchSpace(query);

  if (signal?.aborted) {
    throw new Error("任务已取消");
  }

  const response: AgentRecallResponse = {
    answer: buildAnswerFromSearchResult(searchResult),
    evidence: buildEvidenceFromSpatialResult(searchResult),
    actions: buildRecallActionsFromSearchResult(searchResult),
  };

  return agentRecallResponseSchema.parse(response);
}
