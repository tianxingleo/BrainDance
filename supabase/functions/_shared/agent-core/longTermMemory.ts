import { z } from "npm:zod@3.25";
import { type SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2";

// --- Schemas ---

const recentSearchEntrySchema = z.object({
  query: z.string(),
  mode: z.string(),
  topResultSummary: z.string(),
  regions: z.array(z.string()).default([]),
  objects: z.array(z.string()).default([]),
  timestamp: z.string(),
});

export const longTermMemorySchema = z.object({
  preferredRegions: z.array(z.string()).max(5).default([]),
  preferredAssetTypes: z.array(z.string()).max(5).default([]),
  preferredTimeRanges: z.array(z.string()).max(3).default([]),
  preferredObjects: z.array(z.string()).max(8).default([]),
  recentSearches: z.array(recentSearchEntrySchema).max(10).default([]),
  searchCount: z.number().int().min(0).default(0),
});

export type LongTermMemory = z.infer<typeof longTermMemorySchema>;
export type RecentSearchEntry = z.infer<typeof recentSearchEntrySchema>;

// --- Read ---

export async function loadLongTermMemory(
  supabase: SupabaseClient,
  userId: string,
): Promise<LongTermMemory | null> {
  const { data, error } = await supabase
    .from("user_long_term_memory")
    .select(
      "preferred_regions, preferred_asset_types, preferred_time_ranges, preferred_objects, recent_searches, search_count",
    )
    .eq("user_id", userId)
    .maybeSingle();

  if (error || !data) return null;

  return {
    preferredRegions: data.preferred_regions ?? [],
    preferredAssetTypes: data.preferred_asset_types ?? [],
    preferredTimeRanges: data.preferred_time_ranges ?? [],
    preferredObjects: data.preferred_objects ?? [],
    recentSearches: data.recent_searches ?? [],
    searchCount: data.search_count ?? 0,
  };
}

// --- Threshold ---

export function shouldPersistLongTermMemory(
  turnCount: number,
  existingMemory: LongTermMemory | null,
  shortTermPrefs: { regions?: string[]; assetTypes?: string[] },
): boolean {
  if (!existingMemory || existingMemory.searchCount === 0) return true;
  if (turnCount > 0 && turnCount % 3 === 0) return true;

  const newRegions = (shortTermPrefs.regions ?? []).filter(
    (r) => !existingMemory.preferredRegions.includes(r),
  );
  if (newRegions.length >= 2) return true;

  const newTypes = (shortTermPrefs.assetTypes ?? []).filter(
    (t) => !existingMemory.preferredAssetTypes.includes(t),
  );
  if (newTypes.length >= 2) return true;

  return false;
}

// --- Write ---

export type LongTermMemoryWriteInput = {
  userId: string;
  currentShortTermPreferences: {
    regions?: string[];
    assetTypes?: string[];
    timeRange?: string | null;
  };
  currentQuery: string;
  responseMode: string;
  topResultSummary: string;
  intentObjects: string[];
  intentRegions: string[];
};

export async function persistLongTermMemory(
  supabase: SupabaseClient,
  input: LongTermMemoryWriteInput,
  existingMemory: LongTermMemory | null,
): Promise<void> {
  const prev = existingMemory ?? {
    preferredRegions: [],
    preferredAssetTypes: [],
    preferredTimeRanges: [],
    preferredObjects: [],
    recentSearches: [],
    searchCount: 0,
  };

  const mergedRegions = mergeFrequencyList(
    prev.preferredRegions,
    input.intentRegions,
    5,
  );
  const mergedAssetTypes = mergeFrequencyList(
    prev.preferredAssetTypes,
    input.currentShortTermPreferences.assetTypes ?? [],
    5,
  );
  const mergedObjects = mergeFrequencyList(
    prev.preferredObjects,
    input.intentObjects,
    8,
  );
  const mergedTimeRanges = input.currentShortTermPreferences.timeRange
    ? mergeFrequencyList(
      prev.preferredTimeRanges,
      [input.currentShortTermPreferences.timeRange],
      3,
    )
    : prev.preferredTimeRanges;

  const newEntry: RecentSearchEntry = {
    query: input.currentQuery.slice(0, 100),
    mode: input.responseMode,
    topResultSummary: input.topResultSummary.slice(0, 150),
    regions: input.intentRegions.slice(0, 3),
    objects: input.intentObjects.slice(0, 5),
    timestamp: new Date().toISOString(),
  };
  const recentSearches = [...prev.recentSearches, newEntry].slice(-10);

  const { error } = await supabase
    .from("user_long_term_memory")
    .upsert(
      {
        user_id: input.userId,
        preferred_regions: mergedRegions,
        preferred_asset_types: mergedAssetTypes,
        preferred_time_ranges: mergedTimeRanges,
        preferred_objects: mergedObjects,
        recent_searches: recentSearches,
        search_count: prev.searchCount + 1,
        last_updated_at: new Date().toISOString(),
      },
      { onConflict: "user_id" },
    );

  if (error) {
    console.error("[LongTermMemory] persist failed:", error.message);
  }
}

// --- Helpers ---

function mergeFrequencyList(
  existing: string[],
  newItems: string[],
  maxSize: number,
): string[] {
  const result = [...existing];
  for (const item of newItems) {
    if (!item) continue;
    const idx = result.indexOf(item);
    if (idx !== -1) result.splice(idx, 1);
    result.unshift(item);
  }
  return result.slice(0, maxSize);
}
