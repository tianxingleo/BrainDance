import type { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2";
import { DynamicStructuredTool } from "npm:@langchain/core@0.3/tools";
import { z } from "npm:zod@3.25";

type ModelAssetRow = {
  id: string;
  scene_id: string;
  user_id: string | null;
  description: string | null;
  display_name: string | null;
  summary_title: string | null;
  objects: string[] | null;
  tags: string[] | null;
  preview_img_path: string | null;
  ply_path: string | null;
  meta_info: Record<string, unknown> | null;
  agent_meta: Record<string, unknown> | null;
  place_id: string | null;
  version_label: string | null;
  created_at: string;
};

type MemoryPoseRow = {
  image_name: string;
  tag: string | null;
  transform_matrix: unknown;
  created_at: string;
};

type RelatedLinkRow = {
  source_model_id: string;
  target_model_id: string;
  relation_type: string;
  score: number | null;
};

type MemoryCollectionRow = {
  id: string;
  user_id: string;
  title: string;
  description: string | null;
  cover_model_id: string | null;
  collection_type: string | null;
  created_at: string;
  updated_at: string;
};

type CollectionItemJoinRow = {
  collection_id: string;
  sort_order: number | null;
  note: string | null;
  model_assets: ModelAssetRow | ModelAssetRow[] | null;
};

type AssetToolRuntimeOptions = {
  userId: string;
  selectedModelIds?: string[];
  allowWrite?: boolean;
};

export type PoseSummary = {
  model_id: string;
  pose_count: number;
  top_tags: string[];
  sample_frames: Array<{
    image_name: string;
    tag: string | null;
    transform_matrix: unknown;
    created_at: string;
  }>;
};

export type RelatedModelSummary = {
  model_id: string;
  scene_id: string;
  display_name: string | null;
  relation_type: string;
  relation_score: number;
  created_at: string;
  place_id: string | null;
  version_label: string | null;
};

export type PlaceVersionsResult = {
  place_id: string | null;
  versions: Array<{
    model_id: string;
    scene_id: string;
    display_name: string | null;
    version_label: string | null;
    created_at: string;
  }>;
};

export type MemoryCollectionSummary = {
  collection: MemoryCollectionRow;
  model_count: number;
  items: Array<{
    model_id: string;
    scene_id: string;
    display_name: string | null;
    created_at: string;
    tags: string[];
    sort_order: number;
    note: string | null;
  }>;
  title_suggestion: string;
  summary: string;
  tag_suggestions: string[];
};

export type StoryContext = {
  title: string;
  model_count: number;
  ordered_models: Array<{
    model_id: string;
    scene_id: string;
    display_name: string | null;
    summary_title: string | null;
    version_label: string | null;
    created_at: string;
    tags: string[];
    objects: string[];
    description: string | null;
  }>;
  timeline_summary: string;
  dominant_tags: string[];
};

export type StoryOutline = {
  title: string;
  outline: string[];
  narration_style: string;
};

export type RecentPlaceTrend = {
  place_id: string | null;
  related_models: string[];
  trend: string;
  pose_counts: number[];
  object_counts: number[];
  summary: string;
};

export type MissingObjectPattern = {
  object_name: string;
  baseline_model_ids: string[];
  target_model_id: string | null;
  missing: boolean;
  summary: string;
};

export type PlaceTimelineSummary = {
  place_id: string | null;
  timeline: Array<{
    model_id: string;
    created_at: string;
    version_label: string | null;
    display_name: string | null;
  }>;
  summary: string;
};

export type MemoryGraphSummary = {
  focus_model_id: string;
  related_model_ids: string[];
  place_id: string | null;
  summary: string;
  key_relationships: string[];
};

const listPlaceVersionsSchema = z.object({
  placeId: z.string().uuid().nullable().optional(),
  modelId: z.string().uuid().nullable().optional(),
  limit: z.number().int().min(1).max(20).default(10),
});

const getPoseSummarySchema = z.object({
  modelId: z.string().uuid(),
  limit: z.number().int().min(1).max(20).default(10),
});

const findRelatedModelsSchema = z.object({
  modelId: z.string().uuid(),
  limit: z.number().int().min(1).max(20).default(8),
});

const createMemoryCollectionSchema = z.object({
  title: z.string().trim().min(1).max(120),
  description: z.string().trim().max(2000).nullable().optional(),
  coverModelId: z.string().uuid().nullable().optional(),
  modelIds: z.array(z.string().uuid()).default([]),
  collectionType: z.enum(["manual", "agent_generated", "timeline", "theme"])
    .default("manual"),
});

const addModelsToCollectionSchema = z.object({
  collectionId: z.string().uuid(),
  modelIds: z.array(z.string().uuid()).min(1).max(50),
});

const summarizeCollectionSchema = z.object({
  collectionId: z.string().uuid(),
});

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

function currentDisplayName(row: Pick<ModelAssetRow, "scene_id" | "display_name">): string {
  return row.display_name?.trim() || row.scene_id;
}

function normalizeAssetRow(row: ModelAssetRow): ModelAssetRow {
  return {
    ...row,
    objects: safeArray(row.objects),
    tags: safeArray(row.tags),
    meta_info: row.meta_info && typeof row.meta_info === "object"
      ? row.meta_info
      : {},
    agent_meta: row.agent_meta && typeof row.agent_meta === "object"
      ? row.agent_meta
      : {},
  };
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
    throw new Error(`检测到超出当前已选模型范围的 ID: ${deniedIds.join(", ")}`);
  }
  return requested;
}

async function fetchModelAssets(
  supabase: SupabaseClient,
  modelIds: string[],
  userId: string,
): Promise<ModelAssetRow[]> {
  if (modelIds.length === 0) {
    return [];
  }

  const { data, error } = await supabase
    .from("model_assets")
    .select(
      "id, scene_id, user_id, description, display_name, summary_title, objects, tags, preview_img_path, ply_path, meta_info, agent_meta, place_id, version_label, created_at",
    )
    .eq("user_id", userId)
    .in("id", modelIds);

  if (error) {
    throw new Error(`读取 model_assets 失败: ${error.message}`);
  }

  return (data ?? []).map((row) => normalizeAssetRow(row as ModelAssetRow));
}

// 检索/查询类工具专用：允许命中当前用户自有或 is_official=true 的模型资产，
// 写工具仍使用 fetchModelAssets 做严格的 user_id 隔离。
async function fetchReadableModelAssets(
  supabase: SupabaseClient,
  modelIds: string[],
  userId: string,
): Promise<ModelAssetRow[]> {
  if (modelIds.length === 0) {
    return [];
  }

  const { data, error } = await supabase
    .from("model_assets")
    .select(
      "id, scene_id, user_id, description, display_name, summary_title, objects, tags, preview_img_path, ply_path, meta_info, agent_meta, place_id, version_label, created_at",
    )
    .or(`user_id.eq.${userId},is_official.eq.true`)
    .in("id", modelIds);

  if (error) {
    throw new Error(`读取 model_assets 失败: ${error.message}`);
  }

  return (data ?? []).map((row) => normalizeAssetRow(row as ModelAssetRow));
}

async function fetchPoseCountMap(
  supabase: SupabaseClient,
  modelIds: string[],
): Promise<Map<string, number>> {
  if (modelIds.length === 0) return new Map();

  // memory_poses 没有 user_id 列，调用方必须确保 modelIds 已经按当前用户过滤过。
  const { data, error } = await supabase
    .from("memory_poses")
    .select("model_id")
    .in("model_id", modelIds);

  if (error) {
    throw new Error(`读取 memory_poses 失败: ${error.message}`);
  }

  const map = new Map<string, number>();
  for (const row of data ?? []) {
    const modelId = typeof row.model_id === "string" ? row.model_id : "";
    if (!modelId) continue;
    map.set(modelId, (map.get(modelId) ?? 0) + 1);
  }
  return map;
}

async function assertUserOwnsModels(
  supabase: SupabaseClient,
  modelIds: string[],
  userId: string,
): Promise<void> {
  if (modelIds.length === 0) return;
  const rows = await fetchModelAssets(supabase, modelIds, userId);
  const owned = new Set(rows.map((row) => row.id));
  const missing = modelIds.filter((id) => !owned.has(id));
  if (missing.length > 0) {
    throw new Error(`模型 ${missing.join(", ")} 不属于当前用户，已拒绝操作`);
  }
}

// 检索/查询类工具专用：允许命中自有或 is_official=true 的官方模型，
// 与 assertUserOwnsModels 的差别在于"读"而非"写"，写场景仍走严格归属校验。
async function assertUserCanReadModels(
  supabase: SupabaseClient,
  modelIds: string[],
  userId: string,
): Promise<void> {
  if (modelIds.length === 0) return;
  const rows = await fetchReadableModelAssets(supabase, modelIds, userId);
  const readable = new Set(rows.map((row) => row.id));
  const missing = modelIds.filter((id) => !readable.has(id));
  if (missing.length > 0) {
    throw new Error(`模型 ${missing.join(", ")} 当前用户不可读取，已拒绝操作`);
  }
}

export async function getPoseSummary(
  supabase: SupabaseClient,
  input: z.infer<typeof getPoseSummarySchema>,
  userId: string,
): Promise<PoseSummary> {
  await assertUserCanReadModels(supabase, [input.modelId], userId);
  // memory_poses 没有 user_id 列，归属已通过 assertUserCanReadModels 校验。
  const { data, error } = await supabase
    .from("memory_poses")
    .select("image_name, tag, transform_matrix, created_at")
    .eq("model_id", input.modelId)
    .order("created_at", { ascending: false })
    .limit(Math.max(input.limit, 10));

  if (error) {
    throw new Error(`读取 memory_poses 失败: ${error.message}`);
  }

  const rows = (data ?? []) as MemoryPoseRow[];
  const tagCounter = new Map<string, number>();
  for (const row of rows) {
    const tag = row.tag?.trim();
    if (!tag) continue;
    tagCounter.set(tag, (tagCounter.get(tag) ?? 0) + 1);
  }

  const topTags = [...tagCounter.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([tag]) => tag);

  return {
    model_id: input.modelId,
    pose_count: rows.length,
    top_tags: topTags,
    sample_frames: rows.slice(0, input.limit).map((row) => ({
      image_name: row.image_name,
      tag: row.tag,
      transform_matrix: row.transform_matrix,
      created_at: row.created_at,
    })),
  };
}

export async function findRelatedModels(
  supabase: SupabaseClient,
  input: z.infer<typeof findRelatedModelsSchema>,
  userId: string,
): Promise<RelatedModelSummary[]> {
  const [base] = await fetchReadableModelAssets(supabase, [input.modelId], userId);
  if (!base) {
    throw new Error(`未找到模型 ${input.modelId}`);
  }

  const rows: RelatedModelSummary[] = [];

  try {
    const { data: links } = await supabase
      .from("related_model_links")
      .select("source_model_id, target_model_id, relation_type, score")
      .or(`source_model_id.eq.${input.modelId},target_model_id.eq.${input.modelId}`)
      .limit(input.limit * 2);

    const linkRows = (links ?? []) as RelatedLinkRow[];
    const targetIds = dedupeStrings(linkRows.map((row) =>
      row.source_model_id === input.modelId ? row.target_model_id : row.source_model_id
    ));
    const linkedAssets = await fetchReadableModelAssets(supabase, targetIds, userId);
    const assetById = new Map(linkedAssets.map((row) => [row.id, row]));

    for (const link of linkRows) {
      const relatedId = link.source_model_id === input.modelId
        ? link.target_model_id
        : link.source_model_id;
      const related = assetById.get(relatedId);
      if (!related) continue;
      rows.push({
        model_id: related.id,
        scene_id: related.scene_id,
        display_name: related.display_name,
        relation_type: link.relation_type,
        relation_score: Number(link.score ?? 0),
        created_at: related.created_at,
        place_id: related.place_id,
        version_label: related.version_label,
      });
    }
  } catch {
    // 迁移未执行到本地环境时退化为启发式搜索，避免工具整体失效。
  }

  let builder = supabase
    .from("model_assets")
    .select(
      "id, scene_id, user_id, description, display_name, summary_title, objects, tags, preview_img_path, ply_path, meta_info, agent_meta, place_id, version_label, created_at",
    )
    .or(`user_id.eq.${userId},is_official.eq.true`)
    .neq("id", input.modelId)
    .limit(Math.max(input.limit * 5, 20));

  if (base.place_id) {
    builder = builder.eq("place_id", base.place_id);
  }

  const { data, error } = await builder.order("created_at", { ascending: false });
  if (error) {
    throw new Error(`查询相关模型失败: ${error.message}`);
  }

  const scored = ((data ?? []) as ModelAssetRow[])
    .map((row) => normalizeAssetRow(row))
    .map((row) => {
      const sharedTags = row.tags!.filter((tag) => base.tags!.includes(tag));
      const sharedObjects = row.objects!.filter((item) => base.objects!.includes(item));
      const samePlace = base.place_id && row.place_id === base.place_id ? 0.45 : 0;
      const score = samePlace +
        Math.min(sharedTags.length * 0.08, 0.24) +
        Math.min(sharedObjects.length * 0.05, 0.15);
      return {
        model_id: row.id,
        scene_id: row.scene_id,
        display_name: row.display_name,
        relation_type: samePlace
          ? "same_place"
          : "metadata_overlap",
        relation_score: Number(score.toFixed(4)),
        created_at: row.created_at,
        place_id: row.place_id,
        version_label: row.version_label,
      };
    })
    .filter((row) => row.relation_score > 0)
    .sort((a, b) => b.relation_score - a.relation_score);

  const merged = new Map<string, RelatedModelSummary>();
  for (const row of [...rows, ...scored]) {
    const existing = merged.get(row.model_id);
    if (!existing || existing.relation_score < row.relation_score) {
      merged.set(row.model_id, row);
    }
  }

  return [...merged.values()].slice(0, input.limit);
}

export async function listPlaceVersions(
  supabase: SupabaseClient,
  input: z.infer<typeof listPlaceVersionsSchema>,
  userId: string,
): Promise<PlaceVersionsResult> {
  let placeId = input.placeId ?? null;

  if (input.modelId && !placeId) {
    const [base] = await fetchReadableModelAssets(supabase, [input.modelId], userId);
    if (!base) {
      throw new Error(`未找到模型 ${input.modelId}`);
    }
    placeId = placeId ?? base.place_id;
  }

  let builder = supabase
    .from("model_assets")
    .select(
      "id, scene_id, user_id, description, display_name, summary_title, objects, tags, preview_img_path, ply_path, meta_info, agent_meta, place_id, version_label, created_at",
    )
    .or(`user_id.eq.${userId},is_official.eq.true`)
    .order("created_at", { ascending: true })
    .limit(input.limit);

  if (placeId) {
    builder = builder.eq("place_id", placeId);
  } else if (input.modelId) {
    builder = builder.eq("id", input.modelId);
  }

  const { data, error } = await builder;
  if (error) {
    throw new Error(`读取版本集合失败: ${error.message}`);
  }

  return {
    place_id: placeId,
    versions: ((data ?? []) as ModelAssetRow[]).map((row) => ({
      model_id: row.id,
      scene_id: row.scene_id,
      display_name: row.display_name?.trim() || null,
      version_label: row.version_label?.trim() || null,
      created_at: row.created_at,
    })),
  };
}

export async function createMemoryCollection(
  supabase: SupabaseClient,
  input: z.infer<typeof createMemoryCollectionSchema>,
  options: AssetToolRuntimeOptions,
): Promise<MemoryCollectionRow> {
  const targetIds = restrictModelIds(
    dedupeStrings([
      ...input.modelIds,
      input.coverModelId ?? "",
    ]).filter(Boolean),
    options.selectedModelIds,
  );
  await assertUserOwnsModels(supabase, targetIds, options.userId);

  const { data, error } = await supabase
    .from("memory_collections")
    .insert({
      user_id: options.userId,
      title: input.title,
      description: input.description ?? null,
      cover_model_id: input.coverModelId ?? targetIds[0] ?? null,
      collection_type: input.collectionType,
    })
    .select("id, user_id, title, description, cover_model_id, collection_type, created_at, updated_at")
    .single();

  if (error) {
    throw new Error(`创建专题失败: ${error.message}`);
  }

  const collection = data as MemoryCollectionRow;
  if (targetIds.length > 0) {
    await addModelsToCollection(supabase, {
      collectionId: collection.id,
      modelIds: targetIds,
    }, options);
  }
  return collection;
}

export async function addModelsToCollection(
  supabase: SupabaseClient,
  input: z.infer<typeof addModelsToCollectionSchema>,
  options: AssetToolRuntimeOptions,
): Promise<{ collection_id: string; added_count: number }> {
  const modelIds = restrictModelIds(input.modelIds, options.selectedModelIds);
  await assertUserOwnsModels(supabase, modelIds, options.userId);
  await assertUserOwnsCollection(supabase, input.collectionId, options.userId);

  const rows = modelIds.map((modelId, index) => ({
    collection_id: input.collectionId,
    model_id: modelId,
    sort_order: index,
  }));

  const { error } = await supabase
    .from("memory_collection_items")
    .upsert(rows, { onConflict: "collection_id,model_id", ignoreDuplicates: false });

  if (error) {
    throw new Error(`添加模型到专题失败: ${error.message}`);
  }

  return {
    collection_id: input.collectionId,
    added_count: modelIds.length,
  };
}

async function assertUserOwnsCollection(
  supabase: SupabaseClient,
  collectionId: string,
  userId: string,
): Promise<void> {
  const { data, error } = await supabase
    .from("memory_collections")
    .select("id, user_id")
    .eq("id", collectionId)
    .maybeSingle();
  if (error) {
    throw new Error(`读取专题失败: ${error.message}`);
  }
  if (!data || data.user_id !== userId) {
    throw new Error(`专题 ${collectionId} 不属于当前用户，已拒绝操作`);
  }
}

export async function summarizeCollection(
  supabase: SupabaseClient,
  input: z.infer<typeof summarizeCollectionSchema>,
  userId: string,
): Promise<MemoryCollectionSummary> {
  const { data: collection, error: collectionError } = await supabase
    .from("memory_collections")
    .select("id, user_id, title, description, cover_model_id, collection_type, created_at, updated_at")
    .eq("id", input.collectionId)
    .single();
  if (collectionError || !collection) {
    throw new Error(`读取专题失败: ${collectionError?.message ?? input.collectionId}`);
  }
  if (collection.user_id !== userId) {
    throw new Error(`专题 ${input.collectionId} 不属于当前用户，已拒绝操作`);
  }

  const { data: items, error: itemError } = await supabase
    .from("memory_collection_items")
    .select(
      "collection_id, sort_order, note, model_assets(id, scene_id, user_id, description, display_name, summary_title, objects, tags, preview_img_path, ply_path, meta_info, agent_meta, place_id, version_label, created_at)",
    )
    .eq("collection_id", input.collectionId)
    .order("sort_order", { ascending: true });
  if (itemError) {
    throw new Error(`读取专题条目失败: ${itemError.message}`);
  }

  const normalizedItems = ((items ?? []) as CollectionItemJoinRow[])
    .map((row) => {
      const asset = Array.isArray(row.model_assets) ? row.model_assets[0] : row.model_assets;
      if (!asset) return null;
      const model = normalizeAssetRow(asset);
      // 防御：如果 join 出来的 model 不属于当前用户，丢弃。
      if (model.user_id && model.user_id !== userId) return null;
      return {
        model_id: model.id,
        scene_id: model.scene_id,
        display_name: model.display_name,
        created_at: model.created_at,
        tags: model.tags ?? [],
        sort_order: row.sort_order ?? 0,
        note: row.note ?? null,
      };
    })
    .filter((row): row is NonNullable<typeof row> => Boolean(row));

  const allTags = normalizedItems.flatMap((item) => item.tags);
  const tagCounter = new Map<string, number>();
  for (const tag of allTags) {
    tagCounter.set(tag, (tagCounter.get(tag) ?? 0) + 1);
  }
  const tagSuggestions = [...tagCounter.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([tag]) => tag);

  const firstDate = normalizedItems[0]?.created_at.slice(0, 10);
  const lastDate = normalizedItems[normalizedItems.length - 1]?.created_at.slice(0, 10);
  const titleSuggestion = tagSuggestions.length > 0
    ? `${tagSuggestions[0]}记忆集`
    : `${collection.title}整理稿`;

  return {
    collection: collection as MemoryCollectionRow,
    model_count: normalizedItems.length,
    items: normalizedItems,
    title_suggestion: titleSuggestion,
    summary: normalizedItems.length > 0
      ? `该专题当前收录 ${normalizedItems.length} 个模型，时间跨度从 ${firstDate} 到 ${lastDate}。高频主题包括 ${tagSuggestions.join("、") || "未标注主题"}。`
      : "该专题目前还没有收录模型。",
    tag_suggestions: tagSuggestions,
  };
}

export async function prepareStoryContext(
  supabase: SupabaseClient,
  input: { modelIds?: string[]; collectionId?: string | null },
  options: AssetToolRuntimeOptions,
): Promise<StoryContext> {
  let modelIds = restrictModelIds(input.modelIds ?? [], options.selectedModelIds);
  let title = "空间记忆导览";

  if (input.collectionId) {
    const summary = await summarizeCollection(
      supabase,
      { collectionId: input.collectionId },
      options.userId,
    );
    modelIds = summary.items.map((item) => item.model_id);
    title = summary.collection.title;
  }

  const rows = (await fetchModelAssets(supabase, modelIds, options.userId))
    .sort((a, b) => a.created_at.localeCompare(b.created_at));
  const tagCounter = new Map<string, number>();
  for (const row of rows) {
    for (const tag of row.tags ?? []) {
      tagCounter.set(tag, (tagCounter.get(tag) ?? 0) + 1);
    }
  }
  const dominantTags = [...tagCounter.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([tag]) => tag);

  return {
    title,
    model_count: rows.length,
    ordered_models: rows.map((row) => ({
      model_id: row.id,
      scene_id: row.scene_id,
      display_name: row.display_name,
      summary_title: row.summary_title,
      version_label: row.version_label,
      created_at: row.created_at,
      tags: row.tags ?? [],
      objects: row.objects ?? [],
      description: row.description,
    })),
    timeline_summary: rows.length > 0
      ? `${rows[0].created_at.slice(0, 10)} 到 ${rows[rows.length - 1].created_at.slice(0, 10)} 共 ${rows.length} 个模型`
      : "暂无模型可用于创作",
    dominant_tags: dominantTags,
  };
}

export function generateStoryOutlineFromContext(
  context: StoryContext,
  query: string,
): StoryOutline {
  const focus = context.dominant_tags[0] ?? "空间变化";
  const outline = [
    `${context.title}的整体回顾`,
    `${focus}相关的核心片段`,
    "关键视角与物体变化",
    "结尾回望与下一步整理建议",
  ];
  if (query.includes("旁白")) {
    outline.push("适合作为旁白收束的一段总结");
  }
  return {
    title: `${context.title}导览大纲`,
    outline,
    narration_style: query.includes("温柔") ? "温柔回忆" : "纪实导览",
  };
}

export async function enqueueCreativeTask(
  supabase: SupabaseClient,
  input: {
    query: string;
    modelIds: string[];
    outline: StoryOutline;
    currentSceneId?: string | null;
    userId: string;
  },
): Promise<{ task_id: string; task_type: string; status: string }> {
  const assets = await fetchModelAssets(supabase, input.modelIds.slice(0, 10), input.userId);
  if (assets.length === 0) {
    throw new Error("当前用户下没有可用于 creative task 的模型");
  }
  const sceneId = input.currentSceneId ?? assets[0]?.scene_id ?? `creative-${Date.now()}`;
  const { data, error } = await supabase
    .from("processing_tasks")
    .insert({
      user_id: input.userId,
      scene_id: sceneId,
      status: "pending",
      task_type: "creative_story",
      task_params: {
        query: input.query,
        selectedModelIds: input.modelIds,
        outline: input.outline,
      },
      display_name: input.outline.title,
      description: `Agent 创建的创作任务：${input.query}`,
    })
    .select("id, status, task_type")
    .single();
  if (error) {
    throw new Error(`创建创作任务失败: ${error.message}`);
  }
  return {
    task_id: String(data.id),
    task_type: String(data.task_type ?? "creative_story"),
    status: String(data.status ?? "pending"),
  };
}

function trendLabel(values: number[]): string {
  if (values.length < 2) return "stable";
  const first = values[0] ?? 0;
  const last = values[values.length - 1] ?? 0;
  if (last > first) return "increasing";
  if (last < first) return "declining";
  return "stable";
}

export async function getRecentPlaceTrend(
  supabase: SupabaseClient,
  input: { modelId: string; lookback?: number; userId: string },
): Promise<RecentPlaceTrend> {
  const versions = await listPlaceVersions(supabase, {
    modelId: input.modelId,
    limit: input.lookback ?? 5,
  }, input.userId);
  const assets = await fetchModelAssets(supabase, versions.versions.map((item) => item.model_id), input.userId);
  const allowedIds = assets.map((row) => row.id);
  const poseCountMap = await fetchPoseCountMap(supabase, allowedIds);
  const objectCounts = assets.map((row) => row.objects?.length ?? 0);
  const poseCounts = assets.map((row) => poseCountMap.get(row.id) ?? 0);
  const trend = trendLabel(objectCounts);
  return {
    place_id: versions.place_id,
    related_models: versions.versions.map((item) => item.model_id),
    trend,
    pose_counts: poseCounts,
    object_counts: objectCounts,
    summary: `最近 ${assets.length} 个版本中，可识别物体数量呈${trend === "declining" ? "下降" : trend === "increasing" ? "上升" : "稳定"}趋势。`,
  };
}

export async function findMissingObjectPattern(
  supabase: SupabaseClient,
  input: { modelId: string; objectName: string; lookback?: number; userId: string },
): Promise<MissingObjectPattern> {
  const versions = await listPlaceVersions(supabase, {
    modelId: input.modelId,
    limit: input.lookback ?? 5,
  }, input.userId);
  const assets = await fetchModelAssets(supabase, versions.versions.map((item) => item.model_id), input.userId);
  const target = assets[assets.length - 1] ?? null;
  const baseline = assets.slice(0, -1).filter((row) => (row.objects ?? []).includes(input.objectName));
  const missing = baseline.length > 0 && Boolean(target) &&
    !(target!.objects ?? []).includes(input.objectName);
  return {
    object_name: input.objectName,
    baseline_model_ids: baseline.map((row) => row.id),
    target_model_id: target?.id ?? null,
    missing,
    summary: missing
      ? `${input.objectName} 在更早的版本里出现过，但在最近版本中未再出现。`
      : `${input.objectName} 没有出现稳定缺失模式。`,
  };
}

export async function summarizePlaceChangeTimeline(
  supabase: SupabaseClient,
  input: { modelId: string; limit?: number; userId: string },
): Promise<PlaceTimelineSummary> {
  const versions = await listPlaceVersions(supabase, {
    modelId: input.modelId,
    limit: input.limit ?? 10,
  }, input.userId);
  return {
    place_id: versions.place_id,
    timeline: versions.versions,
    summary: versions.versions.length > 1
      ? `该地点共有 ${versions.versions.length} 个版本，时间从 ${versions.versions[0].created_at.slice(0, 10)} 延续到 ${versions.versions[versions.versions.length - 1].created_at.slice(0, 10)}。`
      : "该地点当前只有一个已知版本。",
  };
}

export async function buildPersonalMemoryGraphSummary(
  supabase: SupabaseClient,
  input: { modelId: string; userId: string },
): Promise<MemoryGraphSummary> {
  const [base] = await fetchModelAssets(supabase, [input.modelId], input.userId);
  if (!base) {
    throw new Error(`未找到模型 ${input.modelId}`);
  }
  const related = await findRelatedModels(supabase, {
    modelId: input.modelId,
    limit: 6,
  }, input.userId);
  const keyRelationships = dedupeStrings([
    base.place_id ? "同一地点版本链" : "",
    ...related.map((item) => item.relation_type),
  ]).filter(Boolean);

  return {
    focus_model_id: base.id,
    related_model_ids: related.map((item) => item.model_id),
    place_id: base.place_id,
    summary: `${currentDisplayName(base)} 当前关联 ${related.length} 个近邻记忆，主要关系包括 ${keyRelationships.join("、") || "弱关联"}。`,
    key_relationships: keyRelationships,
  };
}

export function buildGetPoseSummaryTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions,
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "get_pose_summary",
    description: "读取模型的 pose 摘要，输出 pose 数、常见 tag 和示例视角。",
    schema: getPoseSummarySchema,
    func: async (input) => {
      const [modelId] = restrictModelIds([input.modelId], options.selectedModelIds);
      const result = await getPoseSummary(supabase, { ...input, modelId }, options.userId);
      return JSON.stringify({
        kind: "pose_summary",
        summary: result,
      });
    },
  });
}

export function buildFindRelatedModelsTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions,
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "find_related_models",
    description: "查找可能属于同一地点或描述相近的模型版本。",
    schema: findRelatedModelsSchema,
    func: async (input) => {
      const [modelId] = restrictModelIds([input.modelId], options.selectedModelIds);
      const rows = await findRelatedModels(supabase, { ...input, modelId }, options.userId);
      return JSON.stringify({
        kind: "related_models",
        rows,
      });
    },
  });
}

export function buildListPlaceVersionsTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions,
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "list_place_versions",
    description: "列出同一地点下的时间顺序版本。",
    schema: listPlaceVersionsSchema,
    func: async (input) => {
      const result = await listPlaceVersions(supabase, input, options.userId);
      return JSON.stringify({
        kind: "place_versions",
        result,
      });
    },
  });
}

export function buildCreateMemoryCollectionTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions,
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "create_memory_collection",
    description: "创建记忆专题，可选立即把选中的模型放入专题。",
    schema: createMemoryCollectionSchema,
    func: async (input) => {
      const collection = await createMemoryCollection(supabase, input, options);
      return JSON.stringify({
        kind: "memory_collection",
        action: "create",
        collection,
      });
    },
  });
}

export function buildAddModelsToCollectionTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions,
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "add_models_to_collection",
    description: "把多个模型加入已存在的记忆专题。",
    schema: addModelsToCollectionSchema,
    func: async (input) => {
      const result = await addModelsToCollection(supabase, input, options);
      return JSON.stringify({
        kind: "memory_collection",
        action: "add_models",
        result,
      });
    },
  });
}

export function buildSummarizeCollectionTool(
  supabase: SupabaseClient,
  options: AssetToolRuntimeOptions,
): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: "summarize_collection",
    description: "读取专题并给出标题建议、摘要和标签建议。",
    schema: summarizeCollectionSchema,
    func: async (input) => {
      const summary = await summarizeCollection(supabase, input, options.userId);
      return JSON.stringify({
        kind: "memory_collection_summary",
        summary,
      });
    },
  });
}

