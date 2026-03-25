import { type SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2";
import { z } from "npm:zod@3.25";
import {
  createAiClient,
  createSupabaseAdminClient,
  getEmbedding,
  safeJsonParse,
  searchModels,
} from "../search-models/shared.ts";
import {
  type TimeCompareResponse,
  timeCompareResponseSchema,
} from "./schemas/response.ts";

type CompareWindow = {
  startTime: string;
  endTime: string;
};

type CompareIntent = {
  searchText: string;
  compareFocus: string | null;
  baselineStartTime: string | null;
  baselineEndTime: string | null;
  targetStartTime: string | null;
  targetEndTime: string | null;
  reasoning: string;
};

type SearchRow = {
  id: string;
  scene_id: string;
  description: string | null;
  ply_path: string | null;
  created_at: string;
  similarity: number;
  user_id: string | null;
  matched_frames: Array<Record<string, unknown>>;
};

type CompareFrame = {
  imageName: string;
  similarity: number;
  transformMatrix: unknown;
  tag: string | null;
};

type CompareSceneSnapshot = {
  sceneId: string;
  modelId: string;
  userId: string | null;
  displayName: string | null;
  description: string | null;
  createdAt: string;
  similarity: number;
  objects: string[];
  tags: string[];
  plyPath: string | null;
  bestFrame: CompareFrame | null;
};

type ToolTraceEntry = TimeCompareResponse["toolTrace"][number];
type CompareDiff = TimeCompareResponse["comparison"]["diff"];

const DEFAULT_BUCKET = "braindance-assets";
const DEFAULT_THRESHOLD = 0.45;
const ONE_DAY_MS = 24 * 60 * 60 * 1000;

const compareIntentSchema = z.object({
  search_text: z.string().min(1),
  compare_focus: z.string().nullable(),
  baseline_start_time: z.string().nullable(),
  baseline_end_time: z.string().nullable(),
  target_start_time: z.string().nullable(),
  target_end_time: z.string().nullable(),
  reasoning: z.string().default(""),
});

function asUtcIso(date: Date): string {
  return date.toISOString().replace(/\.\d{3}Z$/, "Z");
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function normalizeIsoOrNull(value: string | null): string | null {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }
  return asUtcIso(parsed);
}

function durationMs(startTime: string, endTime: string): number {
  return Math.max(
    ONE_DAY_MS,
    new Date(endTime).getTime() - new Date(startTime).getTime(),
  );
}

function uniqueStrings(values: unknown): string[] {
  if (!Array.isArray(values)) return [];
  const deduped = new Set<string>();
  for (const item of values) {
    if (typeof item !== "string") continue;
    const trimmed = item.trim();
    if (!trimmed) continue;
    deduped.add(trimmed);
  }
  return [...deduped];
}

function summarizeToolResult(
  toolName: string,
  hit: CompareSceneSnapshot | null,
): string {
  return hit
    ? `${toolName} 命中 ${hit.displayName ?? hit.sceneId}`
    : `${toolName} 未命中可信候选`;
}

function publicUrlForPath(
  supabase: SupabaseClient,
  bucket: string,
  path: string | null,
): string | null {
  if (!path) return null;
  if (/^https?:\/\//.test(path)) return path;
  return supabase.storage.from(bucket).getPublicUrl(path).data.publicUrl;
}

function derivePosesPath(scene: CompareSceneSnapshot): string | null {
  if (!scene.userId || !scene.sceneId) return null;
  return `${scene.userId}/${scene.sceneId}/output/webgl_poses.json`;
}

function normalizeFrame(
  frame: Record<string, unknown>,
  tag: string | null,
): CompareFrame {
  return {
    imageName: typeof frame.image_name === "string" ? frame.image_name : "",
    similarity: clamp(Number(frame.similarity ?? 0), 0, 1),
    transformMatrix: frame.transform_matrix ?? null,
    tag,
  };
}

export function normalizeCompareWindows(
  intent: CompareIntent,
  now = new Date(),
): { baseline: CompareWindow; target: CompareWindow } {
  const baselineStart = normalizeIsoOrNull(intent.baselineStartTime);
  const baselineEnd = normalizeIsoOrNull(intent.baselineEndTime);
  const targetStart = normalizeIsoOrNull(intent.targetStartTime);
  const targetEnd = normalizeIsoOrNull(intent.targetEndTime);

  if (baselineStart && baselineEnd && targetStart && targetEnd) {
    return {
      baseline: { startTime: baselineStart, endTime: baselineEnd },
      target: { startTime: targetStart, endTime: targetEnd },
    };
  }

  if (targetStart && targetEnd) {
    const size = durationMs(targetStart, targetEnd);
    return {
      baseline: {
        startTime: asUtcIso(new Date(new Date(targetStart).getTime() - size)),
        endTime: targetStart,
      },
      target: { startTime: targetStart, endTime: targetEnd },
    };
  }

  if (baselineStart && baselineEnd) {
    const size = durationMs(baselineStart, baselineEnd);
    return {
      baseline: { startTime: baselineStart, endTime: baselineEnd },
      target: {
        startTime: asUtcIso(new Date(now.getTime() - size)),
        endTime: asUtcIso(now),
      },
    };
  }

  const targetWindowEnd = asUtcIso(now);
  const targetWindowStart = asUtcIso(new Date(now.getTime() - 7 * ONE_DAY_MS));
  return {
    baseline: {
      startTime: asUtcIso(new Date(now.getTime() - 14 * ONE_DAY_MS)),
      endTime: targetWindowStart,
    },
    target: {
      startTime: targetWindowStart,
      endTime: targetWindowEnd,
    },
  };
}

export function buildCompareDiff(
  baseline: CompareSceneSnapshot | null,
  target: CompareSceneSnapshot | null,
): CompareDiff {
  const baselineObjects = new Set(baseline?.objects ?? []);
  const targetObjects = new Set(target?.objects ?? []);
  const baselineTags = new Set(baseline?.tags ?? []);
  const targetTags = new Set(target?.tags ?? []);
  const limitations = [
    "当前差分基于搜索命中、描述、objects 与 tags，不是跨扫描几何配准后的严格变化检测。",
  ];

  if (!baseline || !target) {
    limitations.push(
      "至少有一个时间窗口没有命中可信场景，本次结论只反映单侧证据。",
    );
  }

  return {
    commonObjects: [...baselineObjects].filter((item) =>
      targetObjects.has(item)
    ),
    addedObjects: [...targetObjects].filter((item) =>
      !baselineObjects.has(item)
    ),
    removedObjects: [...baselineObjects].filter((item) =>
      !targetObjects.has(item)
    ),
    commonTags: [...baselineTags].filter((item) => targetTags.has(item)),
    addedTags: [...targetTags].filter((item) => !baselineTags.has(item)),
    removedTags: [...baselineTags].filter((item) => !targetTags.has(item)),
    limitations,
  };
}

function formatDate(dateTime: string): string {
  return dateTime.slice(0, 10);
}

function buildAnswer(input: {
  query: string;
  baseline: CompareSceneSnapshot | null;
  target: CompareSceneSnapshot | null;
  diff: CompareDiff;
}): string {
  const { query, baseline, target, diff } = input;
  if (!baseline && !target) {
    return `当前没有在两个时间窗口里找到与“${query}”相关的可信场景，因此还不能给出时间对比结论。`;
  }

  if (!baseline || !target) {
    const only = target ?? baseline;
    return `目前只在单侧时间窗口命中了相关场景 ${
      only?.displayName ?? only?.sceneId ?? "未知场景"
    }，还不能形成双侧时间对比。`;
  }

  const sentences = [
    `已对比 ${formatDate(baseline.createdAt)} 与 ${
      formatDate(target.createdAt)
    } 的两次相关场景记录。`,
  ];

  if (diff.addedObjects.length > 0) {
    sentences.push(
      `较晚时间窗口新增的对象线索有 ${diff.addedObjects.join("、")}。`,
    );
  }
  if (diff.removedObjects.length > 0) {
    sentences.push(
      `较早窗口出现但较晚窗口未再命中的对象有 ${
        diff.removedObjects.join("、")
      }。`,
    );
  }
  if (diff.addedTags.length > 0) {
    sentences.push(`较晚窗口新增的标签线索有 ${diff.addedTags.join("、")}。`);
  }
  if (
    diff.addedObjects.length === 0 &&
    diff.removedObjects.length === 0 &&
    diff.addedTags.length === 0 &&
    diff.removedTags.length === 0
  ) {
    sentences.push("从当前结构化元数据看，没有看到明显的差异信号。");
  }
  sentences.push(
    "结论基于候选场景检索与元数据差分，不代表已经完成同一空间的精确对齐。",
  );
  return sentences.join("");
}

async function parseCompareIntent(
  query: string,
): Promise<CompareIntent> {
  const apiKey = Deno.env.get("DASHSCOPE_API_KEY");
  if (!apiKey) {
    throw new Error("未配置 DASHSCOPE_API_KEY");
  }

  const aiClient = createAiClient(apiKey);
  const today = new Date().toISOString().slice(0, 10);
  const systemPrompt =
    `你是 BrainDance 时间对比意图解析器。当前日期是 ${today}。
你需要从用户输入中提取：
1. search_text：去掉时间词后，用于检索的主题。
2. compare_focus：对比焦点，可为空。
3. baseline_start_time / baseline_end_time：较早窗口的 UTC ISO8601 时间。
4. target_start_time / target_end_time：较晚窗口的 UTC ISO8601 时间。
5. reasoning：用一句话解释你的时间理解。

如果用户只给出一个时间窗口，允许另一侧留空，由系统后处理补全。
只返回 JSON，不要输出任何其他文字。`;

  try {
    const response = await aiClient.chat.completions.create({
      model: Deno.env.get("DASHSCOPE_CHAT_MODEL") ?? "qwen-turbo",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: query },
      ],
      response_format: { type: "json_object" },
    });
    const content = response.choices?.[0]?.message?.content ?? null;
    const parsed = compareIntentSchema.parse(safeJsonParse(content));
    return {
      searchText: parsed.search_text,
      compareFocus: parsed.compare_focus,
      baselineStartTime: parsed.baseline_start_time,
      baselineEndTime: parsed.baseline_end_time,
      targetStartTime: parsed.target_start_time,
      targetEndTime: parsed.target_end_time,
      reasoning: parsed.reasoning,
    };
  } catch (error) {
    console.error("[TimeCompareAgent] 意图解析失败，回退到默认窗口:", error);
    return {
      searchText: query,
      compareFocus: null,
      baselineStartTime: null,
      baselineEndTime: null,
      targetStartTime: null,
      targetEndTime: null,
      reasoning: "意图解析失败，已回退到默认双时间窗口。",
    };
  }
}

async function fetchDisplayNameMap(
  supabase: SupabaseClient,
  sceneIds: string[],
): Promise<Map<string, string>> {
  const map = new Map<string, string>();
  if (sceneIds.length === 0) return map;

  const { data } = await supabase
    .from("processing_tasks")
    .select("scene_id, display_name")
    .in("scene_id", sceneIds);

  for (const row of data ?? []) {
    const sceneId = typeof row.scene_id === "string" ? row.scene_id : "";
    const displayName = typeof row.display_name === "string"
      ? row.display_name.trim()
      : "";
    if (sceneId && displayName) {
      map.set(sceneId, displayName);
    }
  }

  return map;
}

async function fetchFrameTagMap(
  supabase: SupabaseClient,
  modelId: string,
  imageNames: string[],
): Promise<Map<string, string | null>> {
  const map = new Map<string, string | null>();
  if (!modelId || imageNames.length === 0) return map;

  const { data } = await supabase
    .from("memory_poses")
    .select("image_name, tag")
    .eq("model_id", modelId)
    .in("image_name", imageNames);

  for (const row of data ?? []) {
    const imageName = typeof row.image_name === "string" ? row.image_name : "";
    if (!imageName) continue;
    map.set(imageName, typeof row.tag === "string" ? row.tag : null);
  }
  return map;
}

async function enrichSnapshots(
  supabase: SupabaseClient,
  rows: SearchRow[],
): Promise<CompareSceneSnapshot[]> {
  if (rows.length === 0) return [];

  const modelIds = rows.map((row) => row.id);
  const sceneIds = rows.map((row) => row.scene_id);
  const displayNameMap = await fetchDisplayNameMap(supabase, sceneIds);
  const metadataMap = new Map<string, Record<string, unknown>>();

  const { data: metadataRows } = await supabase
    .from("model_assets")
    .select(
      "id, scene_id, user_id, description, objects, tags, ply_path, created_at",
    )
    .in("id", modelIds);

  for (const row of metadataRows ?? []) {
    if (typeof row.id === "string") {
      metadataMap.set(row.id, row as Record<string, unknown>);
    }
  }

  const snapshots: CompareSceneSnapshot[] = [];
  for (const row of rows) {
    const metadata = metadataMap.get(row.id) ?? {};
    const imageNames = Array.isArray(row.matched_frames)
      ? row.matched_frames.map((frame) =>
        typeof frame.image_name === "string" ? frame.image_name : ""
      ).filter((name) => name.length > 0)
      : [];
    const frameTagMap = await fetchFrameTagMap(supabase, row.id, imageNames);
    const frames = Array.isArray(row.matched_frames)
      ? row.matched_frames.map((frame) =>
        normalizeFrame(
          frame,
          frameTagMap.get(
            typeof frame.image_name === "string" ? frame.image_name : "",
          ) ?? null,
        )
      ).filter((frame) => frame.imageName.length > 0)
      : [];
    const bestFrame = frames[0] ?? null;
    const tagValues = [
      ...uniqueStrings(metadata.tags),
      ...frames.map((frame) => frame.tag).filter((tag): tag is string =>
        Boolean(tag)
      ),
    ];

    snapshots.push({
      sceneId: row.scene_id,
      modelId: row.id,
      userId: typeof metadata.user_id === "string"
        ? metadata.user_id
        : row.user_id,
      displayName: displayNameMap.get(row.scene_id) ?? null,
      description: typeof metadata.description === "string"
        ? metadata.description
        : row.description,
      createdAt: typeof metadata.created_at === "string"
        ? metadata.created_at
        : row.created_at,
      similarity: clamp(Number(row.similarity ?? 0), 0, 1),
      objects: uniqueStrings(metadata.objects),
      tags: uniqueStrings(tagValues),
      plyPath: typeof metadata.ply_path === "string"
        ? metadata.ply_path
        : row.ply_path,
      bestFrame,
    });
  }

  return snapshots;
}

async function searchBestSceneInWindow(input: {
  supabase: SupabaseClient;
  queryEmbedding: number[];
  threshold: number;
  window: CompareWindow;
}): Promise<CompareSceneSnapshot | null> {
  const rows = await searchModels(
    input.supabase,
    input.queryEmbedding,
    input.threshold,
    5,
    input.window.startTime,
    input.window.endTime,
  );
  const normalizedRows = Array.isArray(rows) ? rows as SearchRow[] : [];
  const enriched = await enrichSnapshots(input.supabase, normalizedRows);
  return enriched[0] ?? null;
}

export function buildCompareActions(input: {
  baseline: CompareSceneSnapshot | null;
  target: CompareSceneSnapshot | null;
  supabase: SupabaseClient;
  bucket: string;
}): TimeCompareResponse["actions"] {
  const actions: TimeCompareResponse["actions"] = [];
  const scenes = [
    { slot: "baseline" as const, scene: input.baseline },
    { slot: "target" as const, scene: input.target },
  ];

  for (const item of scenes) {
    if (!item.scene) continue;

    actions.push({
      type: "open_scene",
      sceneId: item.scene.sceneId,
      slot: item.slot,
      modelId: item.scene.modelId,
      ply: publicUrlForPath(input.supabase, input.bucket, item.scene.plyPath),
      poses: publicUrlForPath(
        input.supabase,
        input.bucket,
        derivePosesPath(item.scene),
      ),
    });

    if (item.scene.bestFrame?.transformMatrix) {
      actions.push({
        type: "fly_to_pose",
        sceneId: item.scene.sceneId,
        slot: item.slot,
        imageName: item.scene.bestFrame.imageName,
        matrix: item.scene.bestFrame.transformMatrix,
      });
    }
  }

  return actions;
}

export async function runTimeCompareAgent(
  query: string,
  threshold = DEFAULT_THRESHOLD,
): Promise<TimeCompareResponse> {
  const apiKey = Deno.env.get("DASHSCOPE_API_KEY");
  if (!apiKey) {
    throw new Error("未配置 DASHSCOPE_API_KEY");
  }

  const intent = await parseCompareIntent(query);
  const aiClient = createAiClient(apiKey);
  const supabase = createSupabaseAdminClient();
  const windows = normalizeCompareWindows(intent);
  const queryEmbedding = await getEmbedding(aiClient, intent.searchText);
  if (!queryEmbedding) {
    throw new Error("向量生成失败");
  }

  const normalizedThreshold = clamp(threshold, 0, 1);
  const baseline = await searchBestSceneInWindow({
    supabase,
    queryEmbedding,
    threshold: normalizedThreshold,
    window: windows.baseline,
  });
  const target = await searchBestSceneInWindow({
    supabase,
    queryEmbedding,
    threshold: normalizedThreshold,
    window: windows.target,
  });
  const diff = buildCompareDiff(baseline, target);
  const actions = buildCompareActions({
    baseline,
    target,
    supabase,
    bucket: Deno.env.get("SUPABASE_ASSET_BUCKET") ?? DEFAULT_BUCKET,
  });
  const toolTrace: ToolTraceEntry[] = [
    {
      toolName: "search_window",
      args: {
        slot: "baseline",
        query: intent.searchText,
        threshold: normalizedThreshold,
        startTime: windows.baseline.startTime,
        endTime: windows.baseline.endTime,
      },
      resultSummary: summarizeToolResult("search_window", baseline),
    },
    {
      toolName: "search_window",
      args: {
        slot: "target",
        query: intent.searchText,
        threshold: normalizedThreshold,
        startTime: windows.target.startTime,
        endTime: windows.target.endTime,
      },
      resultSummary: summarizeToolResult("search_window", target),
    },
  ];

  return timeCompareResponseSchema.parse({
    success: true,
    intent: {
      originalQuery: query,
      parsedSearchText: intent.searchText,
      compareFocus: intent.compareFocus,
      baseline: windows.baseline,
      target: windows.target,
      reasoning: intent.reasoning,
    },
    comparison: {
      baseline: baseline
        ? {
          sceneId: baseline.sceneId,
          modelId: baseline.modelId,
          displayName: baseline.displayName,
          description: baseline.description,
          createdAt: baseline.createdAt,
          similarity: baseline.similarity,
          objects: baseline.objects,
          tags: baseline.tags,
          matchedFrames: baseline.bestFrame ? [baseline.bestFrame] : [],
          ply: publicUrlForPath(
            supabase,
            Deno.env.get("SUPABASE_ASSET_BUCKET") ?? DEFAULT_BUCKET,
            baseline.plyPath,
          ),
          poses: publicUrlForPath(
            supabase,
            Deno.env.get("SUPABASE_ASSET_BUCKET") ?? DEFAULT_BUCKET,
            derivePosesPath(baseline),
          ),
        }
        : null,
      target: target
        ? {
          sceneId: target.sceneId,
          modelId: target.modelId,
          displayName: target.displayName,
          description: target.description,
          createdAt: target.createdAt,
          similarity: target.similarity,
          objects: target.objects,
          tags: target.tags,
          matchedFrames: target.bestFrame ? [target.bestFrame] : [],
          ply: publicUrlForPath(
            supabase,
            Deno.env.get("SUPABASE_ASSET_BUCKET") ?? DEFAULT_BUCKET,
            target.plyPath,
          ),
          poses: publicUrlForPath(
            supabase,
            Deno.env.get("SUPABASE_ASSET_BUCKET") ?? DEFAULT_BUCKET,
            derivePosesPath(target),
          ),
        }
        : null,
      diff,
    },
    answer: buildAnswer({
      query: intent.searchText,
      baseline,
      target,
      diff,
    }),
    actions,
    toolTrace,
  });
}
