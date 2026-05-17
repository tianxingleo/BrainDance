import type { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2";
import { DynamicStructuredTool } from "npm:@langchain/core@0.3/tools";
import { z } from "npm:zod@3.25";
import type {
  MemoryCollectionSummary,
  PlaceVersionsResult,
  PoseSummary,
  RelatedModelSummary,
} from "./memoryTools.ts";

type ModelAssetRow = {
  id: string;
  scene_id: string;
  description: string | null;
  display_name: string | null;
  objects: string[] | null;
  tags: string[] | null;
  preview_img_path: string | null;
  ply_path: string | null;
  meta_info: Record<string, unknown> | null;
  created_at: string;
};

type MemoryPoseCountRow = {
  model_id: string;
};

export type ListedModelAsset = {
  id: string;
  scene_id: string;
  display_name: string | null;
  description: string | null;
  tags: string[];
  created_at: string;
  preview_img_path: string | null;
  ply_path: string | null;
};

export type ModelAssetBundle = {
  id: string;
  scene_id: string;
  display_name: string | null;
  description: string | null;
  objects: string[];
  tags: string[];
  created_at: string;
  preview_img_path: string | null;
  ply_path: string | null;
  meta_info: Record<string, unknown>;
  pose_count: number;
};

export type CompareModelAssetsResult = {
  rows: ModelAssetBundle[];
  diff: {
    common_tags: string[];
    common_objects: string[];
    tag_only_by_model: Record<string, string[]>;
    object_only_by_model: Record<string, string[]>;
    time_order: string[];
    pose_count_by_model: Record<string, number>;
  };
};

export type DuplicateModelNameGroup = {
  display_name: string;
  count: number;
  rows: ListedModelAsset[];
};

export type AssetOperationResult = {
  tool_name: string;
  dry_run: boolean;
  requires_confirmation: boolean;
  affected_count: number;
  preview: Array<{
    model_id: string;
    scene_id: string;
    old_display_name: string | null;
    new_display_name: string | null;
    old_description: string | null;
    new_description: string | null;
    old_tags: string[];
    new_tags: string[];
  }>;
};

export type AssetToolState = {
  lastToolName: string | null;
  list: ListedModelAsset[] | null;
  bundle: ModelAssetBundle[] | null;
  comparison: CompareModelAssetsResult | null;
  duplicateNames: DuplicateModelNameGroup[] | null;
  operation: AssetOperationResult | null;
  poseSummary: PoseSummary | null;
  relatedModels: RelatedModelSummary[] | null;
  placeVersions: PlaceVersionsResult | null;
  collectionSummary: MemoryCollectionSummary | null;
  threadGrouping: {
    model_ids: string[];
    place_id: string;
    memory_thread_id: string;
  } | null;
};

type AssetToolRuntimeOptions = {
  selectedModelIds?: string[];
  allowWrite?: boolean;
  embeddings?: {
    embedQuery(text: string): Promise<number[]>;
  };
};

const readModelAssetsSchema = z.object({
  mode: z.enum(["list", "duplicate_display_name"]).default("list"),
  modelIds: z.array(z.string().uuid()).default([]),
  sceneIds: z.array(z.string().min(1)).default([]),
  tags: z.array(z.string().min(1)).default([]),
  query: z.string().default(""),
  startTime: z.string().datetime({ offset: true }).nullable().default(null),
  endTime: z.string().datetime({ offset: true }).nullable().default(null),
  limit: z.number().int().min(1).max(50).default(10),
});

const renameModelAssetSchema = z.object({
  modelId: z.string().uuid(),
  newName: z.string().trim().min(1).max(120),
  dryRun: z.boolean().default(true),
});

const batchPatchSchema = z.object({
  modelIds: z.array(z.string().uuid()).min(1),
  patch: z.object({
    displayNameTemplate: z.string().trim().min(1).max(200).optional(),
    displayNamePrefix: z.string().trim().max(120).optional(),
    displayNameSuffix: z.string().trim().max(120).optional(),
    tagsAdd: z.array(z.string().trim().min(1)).default([]),
    tagsRemove: z.array(z.string().trim().min(1)).default([]),
    descriptionReplace: z.string().trim().max(2000).optional(),
    descriptionAppend: z.string().trim().max(2000).optional(),
  }).refine((patch) =>
    Boolean(
      patch.displayNameTemplate ||
        patch.displayNamePrefix ||
        patch.displayNameSuffix ||
        patch.tagsAdd.length > 0 ||
        patch.tagsRemove.length > 0 ||
        patch.descriptionReplace ||
        patch.descriptionAppend,
    ), {
    message: "patch 至少要包含一个可修改字段",
  }),
  dryRun: z.boolean().default(true),
});

const writeModelAssetsSchema = z.object({
  updates: z.array(z.object({
    modelId: z.string().uuid(),
    displayName: z.string().trim().min(1).max(120).optional(),
    description: z.string().trim().max(2000).optional(),
    tags: z.array(z.string().trim().min(1)).optional(),
  }).refine((item) =>
    Boolean(item.displayName || item.description !== undefined || item.tags), {
    message: "每条更新至少包含一个可写字段",
  })).min(1).max(20),
  dryRun: z.boolean().default(true),
});

const getBundleSchema = z.object({
  modelIds: z.array(z.string().uuid()).min(1).max(20),
});

const compareModelAssetsSchema = z.object({
  modelIds: z.array(z.string().uuid()).min(2).max(20),
  fields: z.array(z.enum([
    "display_name",
    "description",
    "objects",
    "tags",
    "created_at",
    "pose_count",
  ])).default([
    "display_name",
    "description",
    "objects",
    "tags",
    "created_at",
    "pose_count",
  ]),
});

const listedModelAssetSchema = z.object({
  id: z.string(),
  scene_id: z.string(),
  display_name: z.string().nullable(),
  description: z.string().nullable(),
  tags: z.array(z.string()),
  created_at: z.string(),
  preview_img_path: z.string().nullable(),
  ply_path: z.string().nullable(),
});

const modelAssetBundleSchema = z.object({
  id: z.string(),
  scene_id: z.string(),
  display_name: z.string().nullable(),
  description: z.string().nullable(),
  objects: z.array(z.string()),
  tags: z.array(z.string()),
  created_at: z.string(),
  preview_img_path: z.string().nullable(),
  ply_path: z.string().nullable(),
  meta_info: z.record(z.string(), z.unknown()),
  pose_count: z.number().int().min(0),
});

const compareModelAssetsResultSchema = z.object({
  rows: z.array(modelAssetBundleSchema),
  diff: z.object({
    common_tags: z.array(z.string()),
    common_objects: z.array(z.string()),
    tag_only_by_model: z.record(z.string(), z.array(z.string())),
    object_only_by_model: z.record(z.string(), z.array(z.string())),
    time_order: z.array(z.string()),
    pose_count_by_model: z.record(z.string(), z.number()),
  }),
});

const duplicateModelNameGroupSchema = z.object({
  display_name: z.string(),
  count: z.number().int().min(2),
  rows: z.array(listedModelAssetSchema),
});

const assetOperationSchema = z.object({
  tool_name: z.string(),
  dry_run: z.boolean(),
  requires_confirmation: z.boolean(),
  affected_count: z.number().int().min(0),
  preview: z.array(z.object({
    model_id: z.string(),
    scene_id: z.string(),
    old_display_name: z.string().nullable(),
    new_display_name: z.string().nullable(),
    old_description: z.string().nullable(),
    new_description: z.string().nullable(),
    old_tags: z.array(z.string()),
    new_tags: z.array(z.string()),
  })),
});

const poseSummaryResultSchema = z.object({
  model_id: z.string(),
  pose_count: z.number().int().min(0),
  top_tags: z.array(z.string()),
  sample_frames: z.array(z.object({
    image_name: z.string(),
    tag: z.string().nullable(),
    transform_matrix: z.custom<unknown>(() => true),
    created_at: z.string(),
  })),
});

const relatedModelSummarySchema = z.object({
  model_id: z.string(),
  scene_id: z.string(),
  display_name: z.string().nullable(),
  relation_type: z.string(),
  relation_score: z.number(),
  created_at: z.string(),
  place_id: z.string().nullable(),
  memory_thread_id: z.string().nullable(),
  version_label: z.string().nullable(),
});

const placeVersionsResultSchema = z.object({
  place_id: z.string().nullable(),
  memory_thread_id: z.string().nullable(),
  versions: z.array(z.object({
    model_id: z.string(),
    scene_id: z.string(),
    display_name: z.string().nullable(),
    version_label: z.string().nullable(),
    created_at: z.string(),
  })),
});

const memoryCollectionSummarySchema = z.object({
  collection: z.object({
    id: z.string(),
    user_id: z.string(),
    title: z.string(),
    description: z.string().nullable(),
    cover_model_id: z.string().nullable(),
    collection_type: z.string().nullable(),
    created_at: z.string(),
    updated_at: z.string(),
  }),
  model_count: z.number().int().min(0),
  items: z.array(z.object({
    model_id: z.string(),
    scene_id: z.string(),
    display_name: z.string().nullable(),
    created_at: z.string(),
    tags: z.array(z.string()),
    sort_order: z.number().int(),
    note: z.string().nullable(),
  })),
  title_suggestion: z.string(),
  summary: z.string(),
  tag_suggestions: z.array(z.string()),
});

const assetToolResultSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("list_model_assets"),
    rows: z.array(listedModelAssetSchema),
  }),
  z.object({
    kind: z.literal("model_asset_bundle"),
    rows: z.array(modelAssetBundleSchema),
  }),
  z.object({
    kind: z.literal("compare_model_assets"),
    rows: compareModelAssetsResultSchema.shape.rows,
    diff: compareModelAssetsResultSchema.shape.diff,
  }),
  z.object({
    kind: z.literal("duplicate_model_names"),
    groups: z.array(duplicateModelNameGroupSchema),
  }),
  z.object({
    kind: z.literal("asset_operation"),
    operation: assetOperationSchema,
  }),
  z.object({
    kind: z.literal("pose_summary"),
    summary: poseSummaryResultSchema,
  }),
  z.object({
    kind: z.literal("related_models"),
    rows: z.array(relatedModelSummarySchema),
  }),
  z.object({
    kind: z.literal("place_versions"),
    result: placeVersionsResultSchema,
  }),
  z.object({
    kind: z.literal("memory_collection_summary"),
    summary: memoryCollectionSummarySchema,
  }),
  z.object({
    kind: z.literal("group_models_into_thread"),
    result: z.object({
      model_ids: z.array(z.string()),
      place_id: z.string(),
      memory_thread_id: z.string(),
    }),
  }),
  z.object({
    kind: z.literal("memory_collection"),
  }),
]);

function safeArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string =>
      typeof item === "string" && item.trim().length > 0
    )
    : [];
}

function dedupeStrings(values: string[]): string[] {
  return [...new Set(values.map((item) => item.trim()).filter(Boolean))];
}

function escapeIlike(value: string): string {
  return value.replace(/[%_,]/g, " ").trim();
}

function normalizeMetaInfo(value: unknown): Record<string, unknown> {
  return value && typeof value === "object"
    ? value as Record<string, unknown>
    : {};
}

function currentDisplayName(
  row: Pick<ModelAssetRow, "scene_id" | "display_name">,
): string {
  return row.display_name?.trim() || row.scene_id;
}

function restrictModelIds(
  requestedIds: string[],
  selectedModelIds?: string[],
): string[] {
  const requested = dedupeStrings(requestedIds);
  const selected = dedupeStrings(selectedModelIds ?? []);

  if (selected.length === 0) {
    return requested;
  }
  if (requested.length === 0) {
    return selected;
  }

  const selectedSet = new Set(selected);
  const deniedIds = requested.filter((id) => !selectedSet.has(id));
  if (deniedIds.length > 0) {
    throw new Error(
      `检测到超出当前已选模型范围的 ID: ${deniedIds.join(", ")}`,
    );
  }
  return requested;
}

export function renderDisplayNameTemplate(
  template: string,
  row: Pick<ModelAssetRow, "scene_id" | "display_name" | "created_at">,
  index: number,
): string {
  const createdDate = row.created_at.slice(0, 10);
  return template
    .replaceAll("{{scene_id}}", row.scene_id)
    .replaceAll("{{current_name}}", currentDisplayName(row))
    .replaceAll("{{created_date}}", createdDate)
    .replaceAll("{{index}}", String(index + 1))
    .trim();
}

async function fetchModelAssets(
  supabase: SupabaseClient,
  modelIds: string[],
): Promise<ModelAssetRow[]> {
  const { data, error } = await supabase
    .from("model_assets")
    .select(
      "id, scene_id, description, display_name, objects, tags, preview_img_path, ply_path, meta_info, created_at",
    )
    .in("id", modelIds)
    .order("created_at", { ascending: true });

  if (error) {
    throw new Error(`读取 model_assets 失败: ${error.message}`);
  }

  return (data ?? []) as ModelAssetRow[];
}

async function fetchSemanticMatchedAssets(input: {
  supabase: SupabaseClient;
  embeddings: { embedQuery(text: string): Promise<number[]> };
  query: string;
  startTime: string | null;
  endTime: string | null;
  limit: number;
  selectedModelIds?: string[];
}): Promise<ModelAssetRow[]> {
  const {
    supabase,
    embeddings,
    query,
    startTime,
    endTime,
    limit,
    selectedModelIds,
  } = input;
  const queryEmbedding = await embeddings.embedQuery(query);
  const { data, error } = await supabase.rpc("match_model_assets", {
    query_embedding: queryEmbedding,
    match_threshold: 0.35,
    match_count: Math.max(limit * 3, 12),
    filter_start: startTime,
    filter_end: endTime,
  } as never) as { data: unknown; error: { message: string } | null };

  if (error) {
    throw new Error(`read_model_assets 语义召回失败: ${error.message}`);
  }

  const rows = Array.isArray(data) ? data as Array<{ id: string }> : [];
  const modelIds = dedupeStrings(rows.map((row) => row.id)).filter(Boolean);
  const allowedIds = selectedModelIds && selectedModelIds.length > 0
    ? new Set(selectedModelIds)
    : null;
  const filteredIds = allowedIds
    ? modelIds.filter((id) => allowedIds.has(id))
    : modelIds;

  if (filteredIds.length === 0) {
    return [];
  }

  const assets = await fetchModelAssets(supabase, filteredIds);
  const order = new Map(filteredIds.map((id, index) => [id, index]));
  return assets
    .sort((left, right) =>
      (order.get(left.id) ?? Number.MAX_SAFE_INTEGER) -
      (order.get(right.id) ?? Number.MAX_SAFE_INTEGER)
    )
    .slice(0, limit);
}

async function fetchPoseCounts(
  supabase: SupabaseClient,
  modelIds: string[],
): Promise<Map<string, number>> {
  if (modelIds.length === 0) {
    return new Map();
  }

  const { data, error } = await supabase
    .from("memory_poses")
    .select("model_id")
    .in("model_id", modelIds);

  if (error) {
    throw new Error(`读取 memory_poses 失败: ${error.message}`);
  }

  const counter = new Map<string, number>();
  for (const row of (data ?? []) as MemoryPoseCountRow[]) {
    counter.set(row.model_id, (counter.get(row.model_id) ?? 0) + 1);
  }
  return counter;
}

async function buildBundle(
  supabase: SupabaseClient,
  modelIds: string[],
): Promise<ModelAssetBundle[]> {
  const [assets, poseCounts] = await Promise.all([
    fetchModelAssets(supabase, modelIds),
    fetchPoseCounts(supabase, modelIds),
  ]);

  const order = new Map(modelIds.map((id, index) => [id, index]));
  return assets
    .map((row) => ({
      id: row.id,
      scene_id: row.scene_id,
      display_name: row.display_name?.trim() || null,
      description: row.description,
      objects: safeArray(row.objects),
      tags: safeArray(row.tags),
      created_at: row.created_at,
      preview_img_path: row.preview_img_path,
      ply_path: row.ply_path,
      meta_info: normalizeMetaInfo(row.meta_info),
      pose_count: poseCounts.get(row.id) ?? 0,
    }))
    .sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));
}

export function buildComparisonResult(
  rows: ModelAssetBundle[],
): CompareModelAssetsResult {
  const commonTags = rows.reduce<string[]>((acc, row, index) => {
    if (index === 0) return [...row.tags];
    return acc.filter((tag) => row.tags.includes(tag));
  }, []);

  const commonObjects = rows.reduce<string[]>((acc, row, index) => {
    if (index === 0) return [...row.objects];
    return acc.filter((item) => row.objects.includes(item));
  }, []);

  const tagOnlyByModel: Record<string, string[]> = {};
  const objectOnlyByModel: Record<string, string[]> = {};
  const poseCountByModel: Record<string, number> = {};

  for (const row of rows) {
    const otherTags = new Set(
      rows.flatMap((item) => item.id === row.id ? [] : item.tags),
    );
    const otherObjects = new Set(
      rows.flatMap((item) => item.id === row.id ? [] : item.objects),
    );
    tagOnlyByModel[row.id] = row.tags.filter((tag) => !otherTags.has(tag));
    objectOnlyByModel[row.id] = row.objects.filter((item) =>
      !otherObjects.has(item)
    );
    poseCountByModel[row.id] = row.pose_count;
  }

  return {
    rows,
    diff: {
      common_tags: commonTags,
      common_objects: commonObjects,
      tag_only_by_model: tagOnlyByModel,
      object_only_by_model: objectOnlyByModel,
      time_order: [...rows]
        .sort((a, b) => a.created_at.localeCompare(b.created_at))
        .map((row) => row.id),
      pose_count_by_model: poseCountByModel,
    },
  };
}

function buildPatchedPreview(
  rows: ModelAssetRow[],
  input: z.infer<typeof batchPatchSchema>,
): AssetOperationResult {
  const preview = rows.map((row, index) => {
    const oldTags = safeArray(row.tags);
    const mergedTags = dedupeStrings([
      ...oldTags.filter((tag) => !input.patch.tagsRemove.includes(tag)),
      ...input.patch.tagsAdd,
    ]);
    const baseName = currentDisplayName(row);
    const templatedName = input.patch.displayNameTemplate
      ? renderDisplayNameTemplate(input.patch.displayNameTemplate, row, index)
      : baseName;
    const nextName = [
      input.patch.displayNamePrefix ?? "",
      templatedName,
      input.patch.displayNameSuffix ?? "",
    ].join("").trim() || null;
    const nextDescription = input.patch.descriptionReplace ?? (
      [
        row.description?.trim() ?? "",
        input.patch.descriptionAppend ?? "",
      ].join(input.patch.descriptionAppend ? "\n" : "").trim() || null
    );

    return {
      model_id: row.id,
      scene_id: row.scene_id,
      old_display_name: row.display_name?.trim() || null,
      new_display_name: nextName,
      old_description: row.description,
      new_description: nextDescription,
      old_tags: oldTags,
      new_tags: mergedTags,
    };
  });

  return {
    tool_name: "batch_patch_model_metadata",
    dry_run: input.dryRun,
    requires_confirmation: input.dryRun,
    affected_count: preview.length,
    preview,
  };
}

async function applyBatchPatch(
  supabase: SupabaseClient,
  rows: ModelAssetRow[],
  operation: AssetOperationResult,
): Promise<void> {
  const nextById = new Map(
    operation.preview.map((item) => [item.model_id, item]),
  );
  for (const row of rows) {
    const next = nextById.get(row.id);
    if (!next) continue;

    const { error } = await supabase
      .from("model_assets")
      .update({
        display_name: next.new_display_name,
        description: next.new_description,
        tags: next.new_tags,
      })
      .eq("id", row.id);

    if (error) {
      throw new Error(`更新模型 ${row.id} 失败: ${error.message}`);
    }
  }
}

function summarizeListRows(rows: ModelAssetRow[]): ListedModelAsset[] {
  return rows.map((row) => ({
    id: row.id,
    scene_id: row.scene_id,
    display_name: row.display_name?.trim() || null,
    description: row.description,
    tags: safeArray(row.tags),
    created_at: row.created_at,
    preview_img_path: row.preview_img_path ?? null,
    ply_path: row.ply_path ?? null,
  }));
}

export function buildAssetAnswer(
  state: AssetToolState,
  options: { query?: string } = {},
): string | null {
  const isRecommendation = isRecommendationQuery(options.query);

  if (state.operation) {
    const actionText = state.operation.dry_run ? "预览" : "执行";
    return `已${actionText} ${state.operation.affected_count} 个模型资产的元数据修改。${
      state.operation.requires_confirmation
        ? "当前仍是 dry run 结果，确认后再执行正式写入。"
        : ""
    }`;
  }
  if (state.comparison) {
    const commonTags = state.comparison.diff.common_tags;
    return `已完成 ${state.comparison.rows.length} 个模型的对比。${
      commonTags.length > 0
        ? `共同标签包括：${commonTags.join("、")}。`
        : "这些模型没有稳定的共同标签。"
    }`;
  }
  if (state.duplicateNames) {
    if (state.duplicateNames.length === 0) {
      return "当前没有发现重名的模型。";
    }
    const details = state.duplicateNames.slice(0, 5).map((group, index) => {
      const scenes = group.rows.slice(0, 3).map((row) => row.scene_id).join("、");
      return `${index + 1}. ${group.display_name}：重复 ${group.count} 次${scenes ? `（scene: ${scenes}）` : ""}`;
    }).join("\n");
    return `当前发现 ${state.duplicateNames.length} 组重名模型：\n${details}\n如果你需要，我可以继续展开这些重名模型的详细摘要。`;
  }
  if (state.bundle) {
    return isRecommendation
      ? buildBundleRecommendationAnswer(state.bundle)
      : buildBundleAnswer(state.bundle);
  }
  if (state.collectionSummary) {
    return `已整理专题“${state.collectionSummary.collection.title}”，当前包含 ${state.collectionSummary.model_count} 个模型。`;
  }
  if (state.relatedModels) {
    return `已找到 ${state.relatedModels.length} 个相关模型版本候选。`;
  }
  if (state.poseSummary) {
    return `已读取模型视角摘要，共 ${state.poseSummary.pose_count} 个 pose。`;
  }
  if (state.placeVersions) {
    return `已列出 ${state.placeVersions.versions.length} 个地点版本。`;
  }
  if (state.threadGrouping) {
    return `已将 ${state.threadGrouping.model_ids.length} 个模型归入同一记忆线程。`;
  }
  if (state.list) {
    return isRecommendation
      ? buildListRecommendationAnswer(state.list)
      : buildListAnswer(state.list);
  }
  return null;
}

function isRecommendationQuery(query?: string): boolean {
  if (!query) {
    return false;
  }
  const normalized = query.replace(/\s+/g, "");
  return /推荐|建议|值得看|先看哪|哪些模型好|有什么模型/.test(normalized);
}

function summarizeRecommendationReason(input: {
  description?: string | null;
  tags: string[];
  poseCount?: number;
}): string {
  if (typeof input.poseCount === "number" && input.poseCount >= 12) {
    return `视角更完整（pose ${input.poseCount} 个）`;
  }
  if (input.tags.length >= 2) {
    return `标签较完整（${input.tags.slice(0, 2).join("、")}）`;
  }
  if (input.description?.trim()) {
    return input.description.trim();
  }
  if (typeof input.poseCount === "number" && input.poseCount > 0) {
    return `已采集 ${input.poseCount} 个 pose`;
  }
  return "信息相对完整，适合优先查看";
}

function rankRecommendationRows<T extends {
  description?: string | null;
  tags: string[];
  created_at: string;
  pose_count?: number;
}>(rows: T[]): T[] {
  return [...rows].sort((left, right) => {
    const leftScore = (left.pose_count ?? 0) * 10 +
      left.tags.length * 3 +
      (left.description?.trim() ? 5 : 0) +
      Date.parse(left.created_at || "1970-01-01T00:00:00Z") / 1_000_000_000_000;
    const rightScore = (right.pose_count ?? 0) * 10 +
      right.tags.length * 3 +
      (right.description?.trim() ? 5 : 0) +
      Date.parse(right.created_at || "1970-01-01T00:00:00Z") / 1_000_000_000_000;
    return rightScore - leftScore;
  });
}

function buildBundleRecommendationAnswer(rows: ModelAssetBundle[]): string {
  if (rows.length === 0) {
    return "当前没有找到可推荐的模型。";
  }

  const recommended = rankRecommendationRows(rows).slice(0, 5);
  const details = recommended.map((row, index) => {
    const name = row.display_name?.trim() || row.scene_id;
    const reason = summarizeRecommendationReason({
      description: row.description,
      tags: row.tags,
      poseCount: row.pose_count,
    });
    return `${index + 1}. ${name}：${reason}`;
  }).join("\n");

  return `我先推荐这 ${recommended.length} 个模型：\n${details}\n如果你想继续缩小范围，我可以再按时间、标签或场景帮你细分。`;
}

function buildListRecommendationAnswer(rows: ListedModelAsset[]): string {
  if (rows.length === 0) {
    return "当前没有找到可推荐的模型。";
  }

  const recommended = rankRecommendationRows(rows).slice(0, 5);
  const details = recommended.map((row, index) => {
    const name = row.display_name?.trim() || row.scene_id;
    const reason = summarizeRecommendationReason({
      description: row.description,
      tags: row.tags,
    });
    return `${index + 1}. ${name}：${reason}`;
  }).join("\n");

  return `我先从当前候选里推荐这 ${recommended.length} 个模型：\n${details}\n如果你需要更精确的推荐，我可以继续读取其中几个模型的详细摘要再细分。`;
}

function buildBundleAnswer(rows: ModelAssetBundle[]): string {
  if (rows.length === 0) {
    return "当前没有读取到可用的模型资产摘要。";
  }

  const displayed = rows.slice(0, 5);
  const intro = displayed.length === 1
    ? "我先整理出 1 个可参考的模型："
    : `我先整理出 ${displayed.length} 个可参考的模型：`;
  const details = displayed.map((row, index) => {
    const name = row.display_name?.trim() || row.scene_id;
    const segments = [
      row.description?.trim() || null,
      row.tags.length > 0 ? `标签：${row.tags.slice(0, 3).join("、")}` : null,
      row.pose_count > 0 ? `pose ${row.pose_count} 个` : null,
    ].filter((item): item is string => Boolean(item));
    return `${index + 1}. ${name}${segments.length > 0 ? `：${segments.join("；")}` : ""}`;
  }).join("\n");
  const suffix = rows.length > displayed.length
    ? `\n（共 ${rows.length} 个，已展示前 ${displayed.length} 个）\n如果你想继续缩小范围，我可以再按时间、标签或场景帮你筛一轮。`
    : "\n如果你想继续缩小范围，我可以再按时间、标签或场景帮你筛一轮。";

  return `${intro}\n${details}${suffix}`;
}

function buildListAnswer(rows: ListedModelAsset[]): string {
  if (rows.length === 0) {
    return "当前没有找到匹配的模型资产。";
  }

  const displayed = rows.slice(0, 5);
  const intro = displayed.length === 1
    ? "当前找到 1 个候选模型："
    : `当前找到 ${displayed.length} 个候选模型：`;
  const details = displayed.map((row, index) => {
    const name = row.display_name?.trim() || row.scene_id;
    const segments = [
      row.description?.trim() || null,
      row.tags.length > 0 ? `标签：${row.tags.slice(0, 3).join("、")}` : null,
    ].filter((item): item is string => Boolean(item));
    return `${index + 1}. ${name}${segments.length > 0 ? `：${segments.join("；")}` : ""}`;
  }).join("\n");
  const suffix = rows.length > displayed.length
    ? `\n（共 ${rows.length} 个，已展示前 ${displayed.length} 个）\n如果你需要，我可以继续读取其中某几个模型的详细摘要。`
    : "\n如果你需要，我可以继续读取其中某几个模型的详细摘要。";

  return `${intro}\n${details}${suffix}`;
}

export function buildReadModelAssetsTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions = {},
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "read_model_assets",
    description:
      "通用模型资产读库工具。可用于按关键词/时间/标签读取资产列表，也可用于重名模型这类只读分析。",
    schema: readModelAssetsSchema,
    func: async (input) => {
      let builder = supabase
        .from("model_assets")
        .select(
          "id, scene_id, description, display_name, objects, tags, preview_img_path, ply_path, meta_info, created_at",
        )
        .order("created_at", { ascending: false })
        .limit(Math.max(input.limit * 5, 20));

      const targetIds = restrictModelIds(
        input.modelIds,
        options.selectedModelIds,
      );
      if (targetIds.length > 0) {
        builder = builder.in("id", targetIds);
      }
      if (input.sceneIds.length > 0) {
        builder = builder.in("scene_id", input.sceneIds);
      }
      if (input.startTime) {
        builder = builder.gte("created_at", input.startTime);
      }
      if (input.endTime) {
        builder = builder.lte("created_at", input.endTime);
      }

      const { data, error } = await builder;
      if (error) {
        throw new Error(`list_model_assets 执行失败: ${error.message}`);
      }

      let rows = (data ?? []) as ModelAssetRow[];
      if (input.tags.length > 0) {
        rows = rows.filter((row) =>
          input.tags.every((tag: string) => safeArray(row.tags).includes(tag))
        );
      }
      if (input.query.trim()) {
        if (options.embeddings) {
          rows = await fetchSemanticMatchedAssets({
            supabase,
            embeddings: options.embeddings,
            query: input.query,
            startTime: input.startTime,
            endTime: input.endTime,
            limit: input.limit,
            selectedModelIds: options.selectedModelIds,
          });
        } else {
          rows = [];
        }
      }

      if (input.mode === "duplicate_display_name") {
        const grouped = new Map<string, ModelAssetRow[]>();
        for (const row of rows) {
          const displayName = row.display_name?.trim();
          if (!displayName) continue;
          const bucket = grouped.get(displayName) ?? [];
          bucket.push(row);
          grouped.set(displayName, bucket);
        }
        return JSON.stringify({
          kind: "duplicate_model_names",
          groups: [...grouped.entries()]
            .map(([displayName, bucket]) => ({
              display_name: displayName,
              count: bucket.length,
              rows: summarizeListRows(
                bucket.sort((left, right) =>
                  right.created_at.localeCompare(left.created_at)
                ),
              ),
            }))
            .filter((group) => group.count >= 2)
            .sort((left, right) => right.count - left.count)
            .slice(0, input.limit),
        });
      }

      return JSON.stringify({
        kind: "list_model_assets",
        rows: summarizeListRows(rows.slice(0, input.limit)),
      });
    },
  });
}

export function buildRenameModelAssetTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions = {},
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "rename_model_asset",
    description:
      "修改单个模型资产的 display_name。默认 dry run，只返回预览，不直接写入。",
    schema: renameModelAssetSchema,
    func: async (input) => {
      const targetIds = restrictModelIds(
        [input.modelId],
        options.selectedModelIds,
      );
      const rows = await fetchModelAssets(supabase, targetIds);
      const row = rows[0];
      if (!row) {
        throw new Error(`未找到模型资产 ${input.modelId}`);
      }
      const dryRun = options.allowWrite === false ? true : input.dryRun;

      const operation: AssetOperationResult = {
        tool_name: "rename_model_asset",
        dry_run: dryRun,
        requires_confirmation: dryRun,
        affected_count: 1,
        preview: [{
          model_id: row.id,
          scene_id: row.scene_id,
          old_display_name: row.display_name?.trim() || null,
          new_display_name: input.newName,
          old_description: row.description,
          new_description: row.description,
          old_tags: safeArray(row.tags),
          new_tags: safeArray(row.tags),
        }],
      };

      if (!dryRun) {
        const { error } = await supabase
          .from("model_assets")
          .update({ display_name: input.newName })
          .eq("id", input.modelId);

        if (error) {
          throw new Error(`rename_model_asset 执行失败: ${error.message}`);
        }
      }

      return JSON.stringify({
        kind: "asset_operation",
        operation,
      });
    },
  });
}

export function buildBatchPatchModelMetadataTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions = {},
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "batch_patch_model_metadata",
    description:
      "批量修改模型资产元数据。仅允许修改 display_name、description、tags，默认 dry run 预览。",
    schema: batchPatchSchema,
    func: async (input) => {
      const targetIds = restrictModelIds(
        input.modelIds,
        options.selectedModelIds,
      );
      const rows = await fetchModelAssets(supabase, targetIds);
      if (rows.length === 0) {
        throw new Error("未找到可修改的模型资产");
      }

      const effectiveInput = {
        ...input,
        dryRun: options.allowWrite === false ? true : input.dryRun,
      };
      const operation = buildPatchedPreview(rows, effectiveInput);
      if (!effectiveInput.dryRun) {
        await applyBatchPatch(supabase, rows, operation);
      }

      return JSON.stringify({
        kind: "asset_operation",
        operation,
      });
    },
  });
}

export function buildWriteModelAssetsTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions = {},
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "write_model_assets",
    description:
      "通用模型资产写库工具。支持一次更新多个模型的 display_name、description、tags；适合“分别改名”“逐条修改”这类请求。",
    schema: writeModelAssetsSchema,
    func: async (input) => {
      const targetIds = restrictModelIds(
        input.updates.map((item: z.infer<typeof writeModelAssetsSchema>["updates"][number]) => item.modelId),
        options.selectedModelIds,
      );
      const rows = await fetchModelAssets(supabase, targetIds);
      if (rows.length === 0) {
        throw new Error("未找到可修改的模型资产");
      }

      const rowById = new Map(rows.map((row) => [row.id, row]));
      const dryRun = options.allowWrite === false ? true : input.dryRun;
      const preview = input.updates.map((
        update: z.infer<typeof writeModelAssetsSchema>["updates"][number],
      ) => {
        const row = rowById.get(update.modelId);
        if (!row) {
          throw new Error(`未找到模型资产 ${update.modelId}`);
        }
        return {
          model_id: row.id,
          scene_id: row.scene_id,
          old_display_name: row.display_name?.trim() || null,
          new_display_name: update.displayName ?? row.display_name?.trim() ?? null,
          old_description: row.description,
          new_description: update.description ?? row.description,
          old_tags: safeArray(row.tags),
          new_tags: update.tags ? dedupeStrings(update.tags) : safeArray(row.tags),
        };
      });

      if (!dryRun) {
        for (const item of preview) {
          const { error } = await supabase
            .from("model_assets")
            .update({
              display_name: item.new_display_name,
              description: item.new_description,
              tags: item.new_tags,
            })
            .eq("id", item.model_id);
          if (error) {
            throw new Error(`write_model_assets 执行失败: ${error.message}`);
          }
        }
      }

      return JSON.stringify({
        kind: "asset_operation",
        operation: {
          tool_name: "write_model_assets",
          dry_run: dryRun,
          requires_confirmation: dryRun,
          affected_count: preview.length,
          preview,
        },
      });
    },
  });
}

export function buildGetModelAssetBundleTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions = {},
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "get_model_asset_bundle",
    description:
      "读取一个或多个模型资产的完整摘要，包括 pose_count，适合给 Agent 做说明和分析。",
    schema: getBundleSchema,
    func: async ({ modelIds }) => {
      const bundle = await buildBundle(
        supabase,
        restrictModelIds(modelIds, options.selectedModelIds),
      );
      return JSON.stringify({
        kind: "model_asset_bundle",
        rows: bundle,
      });
    },
  });
}

export function buildCompareModelAssetsTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions = {},
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "compare_model_assets",
    description:
      "对多个模型资产做结构化对比，输出共同标签、差异对象、时间顺序和 pose_count。",
    schema: compareModelAssetsSchema,
    func: async ({ modelIds, fields }) => {
      const bundle = await buildBundle(
        supabase,
        restrictModelIds(modelIds, options.selectedModelIds),
      );
      const comparison = buildComparisonResult(bundle);
      const requestedFields = new Set(fields);
      return JSON.stringify({
        kind: "compare_model_assets",
        rows: comparison.rows,
        selected_rows: comparison.rows.map((row) => ({
          id: row.id,
          scene_id: row.scene_id,
          display_name: row.display_name,
          description: requestedFields.has("description")
            ? row.description
            : undefined,
          objects: requestedFields.has("objects") ? row.objects : undefined,
          tags: requestedFields.has("tags") ? row.tags : undefined,
          created_at: requestedFields.has("created_at")
            ? row.created_at
            : undefined,
          pose_count: requestedFields.has("pose_count")
            ? row.pose_count
            : undefined,
        })),
        selected_fields: fields,
        diff: comparison.diff,
      });
    },
  });
}

export function createEmptyAssetToolState(): AssetToolState {
  return {
    lastToolName: null,
    list: null,
    bundle: null,
    comparison: null,
    duplicateNames: null,
    operation: null,
    poseSummary: null,
    relatedModels: null,
    placeVersions: null,
    collectionSummary: null,
    threadGrouping: null,
  };
}

export function collectAssetToolResult(
  toolName: string,
  payload: string,
  state: AssetToolState,
): number {
  const parsed = assetToolResultSchema.parse(JSON.parse(payload));
  state.lastToolName = toolName;

  if (parsed.kind === "list_model_assets") {
    if (!state.list) {
      state.list = parsed.rows;
    } else {
      const seen = new Set(state.list.map((r) => r.id));
      const newRows = parsed.rows.filter((r) => !seen.has(r.id));
      state.list = [...state.list, ...newRows];
    }
    return state.list.length;
  }
  if (parsed.kind === "model_asset_bundle") {
    state.bundle = parsed.rows;
    return state.bundle.length;
  }
  if (parsed.kind === "compare_model_assets") {
    state.comparison = {
      rows: parsed.rows,
      diff: parsed.diff,
    };
    return state.comparison.rows.length;
  }
  if (parsed.kind === "duplicate_model_names") {
    state.duplicateNames = parsed.groups;
    return state.duplicateNames.length;
  }
  if (parsed.kind === "asset_operation" && parsed.operation) {
    state.operation = parsed.operation;
    return state.operation.affected_count;
  }
  if (parsed.kind === "pose_summary" && parsed.summary) {
    state.poseSummary = parsed.summary as PoseSummary;
    return parsed.summary.pose_count;
  }
  if (parsed.kind === "related_models") {
    state.relatedModels = parsed.rows;
    return state.relatedModels.length;
  }
  if (parsed.kind === "place_versions" && parsed.result) {
    state.placeVersions = parsed.result;
    return state.placeVersions.versions.length;
  }
  if (parsed.kind === "memory_collection_summary" && parsed.summary) {
    state.collectionSummary = parsed.summary;
    return state.collectionSummary.model_count;
  }
  if (parsed.kind === "group_models_into_thread" && parsed.result) {
    state.threadGrouping = parsed.result;
    return state.threadGrouping?.model_ids.length ?? 0;
  }
  if (parsed.kind === "memory_collection") {
    return 1;
  }

  return 0;
}
