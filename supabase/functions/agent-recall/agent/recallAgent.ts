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
  const displayName = typeof topResult.display_name === "string"
    ? topResult.display_name.trim()
    : "";
  const sceneId = displayName.length > 0
    ? displayName
    : typeof topResult.scene_id === "string"
    ? topResult.scene_id
    : "未知场景";
  const similarity = Math.round(Number(topResult.similarity ?? 0) * 100);
  return `已找到最相关的空间记忆，命中场景 ${sceneId}，相关描述为“${description}”，相似度约 ${similarity}%。`;
}

export async function runRecallAgent(
  query: string,
): Promise<AgentRecallResponse> {
  const searchResult = await searchSpace(query);
  const response: AgentRecallResponse = {
    answer: buildAnswerFromSearchResult(searchResult),
    evidence: buildEvidenceFromSpatialResult(searchResult),
    actions: buildRecallActionsFromSearchResult(searchResult),
  };

  return agentRecallResponseSchema.parse(response);
}
