import {
  runSpatialSearchAgent,
  type SpatialSearchResponse,
} from "../../spatial-search-agent/agent.ts";

export async function searchSpace(
  query: string,
): Promise<SpatialSearchResponse> {
  return await runSpatialSearchAgent(query);
}
