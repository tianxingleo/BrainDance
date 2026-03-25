import { searchSpace } from "../tools/searchSpace.ts";
import { buildEvidenceFromSpatialResult } from "../tools/getSceneAsset.ts";
import { mapSpatialActionsToRecallActions } from "../tools/buildViewAction.ts";
import {
  type AgentRecallResponse,
  agentRecallResponseSchema,
} from "../schemas/response.ts";

export async function runRecallAgent(
  query: string,
): Promise<AgentRecallResponse> {
  const spatialResult = await searchSpace(query);
  const response: AgentRecallResponse = {
    answer: spatialResult.answer,
    evidence: buildEvidenceFromSpatialResult(spatialResult),
    actions: mapSpatialActionsToRecallActions(spatialResult.actions),
  };

  return agentRecallResponseSchema.parse(response);
}
