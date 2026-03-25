import {
  runSearchModelsQuery,
  type SearchModelsResponse,
} from "../../search-models/shared.ts";

export async function searchSpace(
  query: string,
): Promise<SearchModelsResponse> {
  return await runSearchModelsQuery(query);
}
