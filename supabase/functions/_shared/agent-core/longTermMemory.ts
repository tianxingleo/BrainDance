import { z } from "npm:zod@3.25";
import { type SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2";

// --- Schemas ---

const recentSearchEntrySchema = z.object({
  query: z.string(),
  mode: z.string(),
  topResultSummary: z.string(),
  regions: z.array(z.string()).default([]),
  objects: z.array(z.string()).default([]),
  assetTypes: z.array(z.string()).default([]),
  timeRanges: z.array(z.string()).default([]),
  confidence: z.number().min(0).max(1).default(0),
  timestamp: z.string(),
});

export const longTermMemorySchema = z.object({
  preferredRegions: z.array(z.string()).max(5).default([]),
  preferredAssetTypes: z.array(z.string()).max(5).default([]),
  preferredTimeRanges: z.array(z.string()).max(3).default([]),
  preferredObjects: z.array(z.string()).max(8).default([]),
  recentSearches: z.array(recentSearchEntrySchema).max(10).default([]),
  searchCount: z.number().int().min(0).default(0),
  lastUpdatedAt: z.string().nullable().optional().default(null),
});

export type LongTermMemory = z.infer<typeof longTermMemorySchema>;
export type RecentSearchEntry = z.infer<typeof recentSearchEntrySchema>;

export type LongTermMemorySignal = {
  query: string;
  responseMode: string;
  topResultSummary: string;
  regions: string[];
  assetTypes: string[];
  timeRanges: string[];
  objects: string[];
  confidence: number;
};

type LongTermMemoryCandidate = {
  description?: string | null;
  tags?: string[] | null;
  objects?: string[] | null;
  display_name?: string | null;
};

type LongTermMemoryIntent = {
  objectHint?: string | null;
  locationHint?: string | null;
  sceneHint?: string | null;
  timeHint?: string | null;
};

type LongTermMemoryResponse = {
  mode: string;
  answer: string;
  intent?: LongTermMemoryIntent | null;
  top_candidates?: LongTermMemoryCandidate[];
  evidence?: {
    description?: string | null;
    tags?: string[] | null;
  } | null;
};

const LOCATION_HINTS = [
  "客厅",
  "卧室",
  "厨房",
  "书房",
  "餐厅",
  "阳台",
  "卫生间",
  "浴室",
  "走廊",
  "门口",
  "书桌",
  "桌面",
  "茶几",
  "沙发",
  "床边",
  "柜子",
  "窗边",
  "窗口",
  "车内",
  "车里",
  "室外",
  "院子",
  "楼梯",
  "入口",
  "墙边",
  "地面",
];

const ASSET_TYPE_PATTERNS: Array<{ type: string; pattern: RegExp }> = [
  { type: "model", pattern: /(\bmodel\b|模型|资产)/i },
  { type: "scene", pattern: /(场景|空间|环境|房间|室内|室外)/ },
  { type: "pose", pattern: /(pose|位姿|姿态|姿势|视角)/i },
  { type: "collection", pattern: /(集合|专题|归档|收藏|编组)/ },
  { type: "timeline", pattern: /(时间线|对比|变化|前后|历史)/ },
  { type: "3dgs", pattern: /(3dgs|高斯|点云|ply|gaussian)/i },
];

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  const result: string[] = [];
  for (const value of values) {
    const normalized = (value ?? "").trim();
    if (!normalized || result.includes(normalized)) continue;
    result.push(normalized);
  }
  return result;
}

function extractLocationHintsFromText(text: string): string[] {
  const hints = LOCATION_HINTS.filter((hint) => text.includes(hint));
  if (hints.length > 0) return hints;

  const match = text.match(/(?:在|到|去|靠近|附近|旁边|周围|周边)([^，。；;、\s]{1,12})/);
  return match?.[1] ? [match[1]] : [];
}

function extractTimeRangesFromText(text: string): string[] {
  const normalized = text.trim();
  if (!normalized) return [];

  if (/(最近|刚才|刚刚|近来)/.test(normalized)) return ["最近"];
  if (/(今天|今日)/.test(normalized)) return ["今天"];
  if (/(昨天|昨日)/.test(normalized)) return ["昨天"];
  if (/(本周|这周|最近一周|近一周)/.test(normalized)) return ["本周"];
  if (/(上周|上个星期|上周)/.test(normalized)) return ["上周"];
  if (/(今年|去年|上个月|本月|这个月)/.test(normalized)) {
    return [normalized.match(/(今年|去年|上个月|本月|这个月)/)?.[1] ?? normalized];
  }

  return [];
}

function extractAssetTypesFromText(text: string): string[] {
  return ASSET_TYPE_PATTERNS
    .filter(({ pattern }) => pattern.test(text))
    .map(({ type }) => type);
}

function buildRegionSignals(
  query: string,
  intent: LongTermMemoryIntent | null | undefined,
  candidate: LongTermMemoryCandidate | null,
  evidenceTags: string[],
): string[] {
  return uniqueStrings([
    intent?.locationHint,
    ...extractLocationHintsFromText(query),
    ...extractLocationHintsFromText(candidate?.display_name ?? ""),
    ...extractLocationHintsFromText(candidate?.description ?? ""),
    ...extractLocationHintsFromText(evidenceTags.join(" ")),
  ]);
}

function buildObjectSignals(
  intent: LongTermMemoryIntent | null | undefined,
  candidate: LongTermMemoryCandidate | null,
  evidenceTags: string[],
): string[] {
  const filteredTags = evidenceTags.filter((tag) =>
    !LOCATION_HINTS.some((hint) => tag.includes(hint))
  );

  return uniqueStrings([
    intent?.objectHint,
    intent?.sceneHint,
    ...(candidate?.objects ?? []),
    ...filteredTags,
  ]).slice(0, 8);
}

function buildTimeSignals(
  query: string,
  intent: LongTermMemoryIntent | null | undefined,
): string[] {
  return uniqueStrings([
    intent?.timeHint,
    ...extractTimeRangesFromText(query),
  ]);
}

function buildAssetTypeSignals(query: string, responseMode: string): string[] {
  const modeSignals = responseMode === "spatial_search"
    ? ["scene"]
    : responseMode === "asset_metadata"
    ? ["model"]
    : responseMode === "time_compare"
    ? ["comparison"]
    : responseMode === "creative"
    ? ["creative"]
    : responseMode === "memory_graph"
    ? ["memory_graph"]
    : [];

  return uniqueStrings([
    ...extractAssetTypesFromText(query),
    ...modeSignals,
  ]);
}

export function buildLongTermMemorySignal(
  query: string,
  response: LongTermMemoryResponse,
): LongTermMemorySignal {
  const topCandidate = response.top_candidates?.[0] ?? null;
  const evidenceTags = uniqueStrings([
    ...(response.evidence?.tags ?? []),
    ...(topCandidate?.tags ?? []),
  ]);

  const regions = buildRegionSignals(query, response.intent, topCandidate, evidenceTags);
  const objects = buildObjectSignals(response.intent, topCandidate, evidenceTags);
  const timeRanges = buildTimeSignals(query, response.intent);
  const assetTypes = buildAssetTypeSignals(query, response.mode);
  const confidence = Math.min(
    1,
    0.25 +
      regions.length * 0.15 +
      objects.length * 0.15 +
      timeRanges.length * 0.2 +
      assetTypes.length * 0.15,
  );

  return {
    query,
    responseMode: response.mode,
    topResultSummary: response.top_candidates?.length > 0
      ? `${response.top_candidates[0]?.description ?? response.top_candidates[0]?.display_name ?? "unknown"}`
      : response.answer.slice(0, 120),
    regions,
    assetTypes,
    timeRanges,
    objects,
    confidence,
  };
}

// --- Read ---

export async function loadLongTermMemory(
  supabase: SupabaseClient,
  userId: string,
): Promise<LongTermMemory | null> {
  const { data, error } = await supabase
    .from("user_long_term_memory")
    .select(
      "preferred_regions, preferred_asset_types, preferred_time_ranges, preferred_objects, recent_searches, search_count, last_updated_at",
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
    lastUpdatedAt: data.last_updated_at ?? null,
  };
}

// --- Threshold ---

export function shouldPersistLongTermMemory(
  _existingMemory: LongTermMemory | null,
  _signal: LongTermMemorySignal,
): boolean {
  return true;
}

// --- Write ---

export type LongTermMemoryWriteInput = {
  userId: string;
  currentQuery: string;
  responseMode: string;
  topResultSummary: string;
  signal: LongTermMemorySignal;
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
    lastUpdatedAt: null,
  };

  const mergedRegions = mergeFrequencyList(
    prev.preferredRegions,
    input.signal.regions,
    5,
  );
  const mergedAssetTypes = mergeFrequencyList(
    prev.preferredAssetTypes,
    input.signal.assetTypes,
    5,
  );
  const mergedObjects = mergeFrequencyList(
    prev.preferredObjects,
    input.signal.objects,
    8,
  );
  const mergedTimeRanges = mergeFrequencyList(
    prev.preferredTimeRanges,
    input.signal.timeRanges,
    3,
  );

  const newEntry: RecentSearchEntry = {
    query: input.currentQuery.slice(0, 100),
    mode: input.responseMode,
    topResultSummary: input.topResultSummary.slice(0, 150),
    regions: input.signal.regions.slice(0, 3),
    objects: input.signal.objects.slice(0, 5),
    assetTypes: input.signal.assetTypes.slice(0, 3),
    timeRanges: input.signal.timeRanges.slice(0, 3),
    confidence: input.signal.confidence,
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
