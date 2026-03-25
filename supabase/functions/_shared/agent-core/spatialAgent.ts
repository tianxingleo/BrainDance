import {
  createClient,
  type SupabaseClient,
} from "https://esm.sh/@supabase/supabase-js@2";
import {
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "npm:@langchain/core@0.3/messages";
import { DynamicStructuredTool } from "npm:@langchain/core@0.3/tools";
import { ChatOpenAI, OpenAIEmbeddings } from "npm:@langchain/openai@0.6";
import { z } from "npm:zod@3.25";
import {
  type AssetToolState,
  buildAssetAnswer,
  buildBatchPatchModelMetadataTool,
  buildCompareModelAssetsTool,
  buildGetModelAssetBundleTool,
  buildListModelAssetsTool,
  buildRenameModelAssetTool,
  collectAssetToolResult,
  type CompareModelAssetsResult,
  createEmptyAssetToolState,
  type ListedModelAsset,
  type ModelAssetBundle,
} from "./assetTools.ts";
import {
  buildAddModelsToCollectionTool,
  buildCreateMemoryCollectionTool,
  buildFindRelatedModelsTool,
  buildGetPoseSummaryTool,
  buildGroupModelsIntoThreadTool,
  buildListPlaceVersionsTool,
  buildSummarizeCollectionTool,
  buildPersonalMemoryGraphSummary,
  enqueueCreativeTask,
  findMissingObjectPattern,
  generateStoryOutlineFromContext,
  getRecentPlaceTrend,
  prepareStoryContext,
  summarizePlaceChangeTimeline,
} from "./memoryTools.ts";
import { runTimeCompareAgent } from "../../time-compare-agent/agent.ts";

export const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

const DEFAULT_DASHSCOPE_BASE_URL =
  "https://dashscope.aliyuncs.com/compatible-mode/v1";
const DEFAULT_CHAT_MODEL = "qwen3.5-plus";
const DEFAULT_EMBEDDING_MODEL = "text-embedding-v2";
const DEFAULT_BUCKET = "braindance-assets";
const MAX_AGENT_TOOL_ROUNDS = 3;
const MIN_AGENT_CANDIDATES = 3;
const MIN_AGENT_TOP_SCORE = 0.62;

export const searchTargetTypeSchema = z.enum([
  "object",
  "location",
  "time",
  "scene",
]);

const spatialIntentSchema = z.object({
  rewrittenQuery: z.string().min(1),
  targetType: searchTargetTypeSchema,
  objectHint: z.string().nullable(),
  locationHint: z.string().nullable(),
  sceneHint: z.string().nullable(),
  timeHint: z.string().nullable(),
  startTime: z.string().datetime({ offset: true }).nullable(),
  endTime: z.string().datetime({ offset: true }).nullable(),
  reasoning: z.string(),
});

const agentModeSchema = z.enum([
  "chat",
  "spatial_search",
  "asset_metadata",
  "time_compare",
  "creative",
  "memory_graph",
]);

const agentRouteSchema = z.object({
  mode: agentModeSchema,
  reasoning: z.string(),
});

const visualizationActionSchema = z.object({
  type: z.enum(["open_scene", "fly_to_pose"]),
  title: z.string(),
  payload: z.record(z.string(), z.unknown()),
});

const selectionSchema = z.object({
  selectedSceneId: z.string().nullable(),
  selectedModelId: z.string().nullable(),
  selectedPoseImageId: z.string().nullable(),
  selectionReason: z.string(),
  confidence: z.number().min(0).max(1),
  answer: z.string(),
  actions: z.array(visualizationActionSchema),
});

type SearchTargetType = z.infer<typeof searchTargetTypeSchema>;
type SpatialIntent = z.infer<typeof spatialIntentSchema>;
type AgentMode = z.infer<typeof agentModeSchema>;
type VisualizationAction = z.infer<typeof visualizationActionSchema>;
type SelectionResult = z.infer<typeof selectionSchema>;

export type SpatialSearchAgentOptions = {
  selectedModelIds?: string[];
  executionMode?: "preview" | "execute";
  currentSceneId?: string | null;
  currentModelId?: string | null;
  currentMode?: "search" | "compare" | "batch_edit" | "collection" | null;
  candidateSceneIds?: string[];
  sessionId?: string;
  conversationSummary?: string | null;
};

type RuntimeEnv = {
  dashscopeApiKey: string;
  dashscopeBaseUrl: string;
  chatModel: string;
  embeddingModel: string;
  supabaseUrl: string;
  supabaseServiceRoleKey: string;
  storageBucket: string;
};

type SceneRow = {
  id: string;
  scene_id: string;
  user_id: string | null;
  description: string | null;
  objects: string[] | null;
  tags: string[] | null;
  ply_path: string | null;
  preview_img_path: string | null;
  meta_info: Record<string, unknown> | null;
  created_at: string;
};

type PoseFrame = {
  image_name: string;
  transform_matrix: number[] | number[][] | null;
  similarity: number;
  tag: string | null;
};

type PoseSearchRow = {
  id: string;
  scene_id: string;
  description: string | null;
  ply_path: string | null;
  created_at: string;
  user_id: string | null;
  similarity: number;
  matched_frames: PoseFrame[];
};

type SceneCandidate = {
  modelId: string;
  sceneId: string;
  userId: string | null;
  description: string;
  objects: string[];
  tags: string[];
  plyPath: string | null;
  previewImgPath: string | null;
  createdAt: string;
  metaInfo: Record<string, unknown>;
  sourceScores: Record<string, number>;
  bestPose: PoseFrame | null;
};

type ToolTraceEntry = {
  toolName: string;
  args: Record<string, unknown>;
  resultSummary: string;
};

type ToolCallLike = {
  id: string;
  name: string;
  args: Record<string, unknown>;
};

export type SpatialSearchResponse = {
  success: true;
  mode:
    | "chat"
    | "spatial_search"
    | "asset_metadata"
    | "time_compare"
    | "creative"
    | "memory_graph";
  intent: SpatialIntent | null;
  selection: {
    scene_id: string | null;
    model_id: string | null;
    pose_image_id: string | null;
    confidence: number;
    reason: string;
  };
  answer: string;
  actions: VisualizationAction[];
  viewer_payload: {
    ply: string | null;
    poses: string | null;
    matrix: number[] | number[][] | null;
    imageId: string | null;
  };
  evidence?: Record<string, unknown> | null;
  candidates: Array<{
    scene_id: string;
    model_id: string;
    score: number;
    description: string;
    pose_image_id: string | null;
  }>;
  top_candidates?: Array<{
    scene_id: string;
    model_id: string;
    score: number;
    description: string;
    pose_image_id: string | null;
  }>;
  selected_candidate_reason?: string | null;
  tool_trace: ToolTraceEntry[];
  asset_context?: {
    last_tool_name: string | null;
    list: ListedModelAsset[] | null;
    bundle: ModelAssetBundle[] | null;
    comparison: CompareModelAssetsResult | null;
    operation: ReturnType<typeof serializeAssetOperation>;
    pose_summary?: AssetToolState["poseSummary"];
    related_models?: AssetToolState["relatedModels"];
    place_versions?: AssetToolState["placeVersions"];
    collection_summary?: AssetToolState["collectionSummary"];
    thread_grouping?: AssetToolState["threadGrouping"];
  };
  compare_context?: Record<string, unknown> | null;
  collection_context?: Record<string, unknown> | null;
  creative_context?: Record<string, unknown> | null;
  memory_graph_context?: Record<string, unknown> | null;
};

function ensureRuntimeEnv(): RuntimeEnv {
  const dashscopeApiKey = Deno.env.get("DASHSCOPE_API_KEY") ?? "";
  const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
  const supabaseServiceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ??
    "";

  if (!dashscopeApiKey) {
    throw new Error("未配置 DASHSCOPE_API_KEY");
  }
  if (!supabaseUrl) {
    throw new Error("未配置 SUPABASE_URL");
  }
  if (!supabaseServiceRoleKey) {
    throw new Error("未配置 SUPABASE_SERVICE_ROLE_KEY");
  }

  return {
    dashscopeApiKey,
    dashscopeBaseUrl: Deno.env.get("DASHSCOPE_BASE_URL") ??
      DEFAULT_DASHSCOPE_BASE_URL,
    chatModel: Deno.env.get("DASHSCOPE_CHAT_MODEL") ?? DEFAULT_CHAT_MODEL,
    embeddingModel: Deno.env.get("DASHSCOPE_EMBEDDING_MODEL") ??
      DEFAULT_EMBEDDING_MODEL,
    supabaseUrl,
    supabaseServiceRoleKey,
    storageBucket: Deno.env.get("SUPABASE_ASSET_BUCKET") ?? DEFAULT_BUCKET,
  };
}

function createSupabaseAdminClient(env: RuntimeEnv): SupabaseClient {
  return createClient(env.supabaseUrl, env.supabaseServiceRoleKey);
}

function createChatModel(env: RuntimeEnv): ChatOpenAI {
  return new ChatOpenAI({
    apiKey: env.dashscopeApiKey,
    model: env.chatModel,
    temperature: 0,
    configuration: {
      baseURL: env.dashscopeBaseUrl,
    },
  });
}

function createEmbeddingsModel(env: RuntimeEnv): OpenAIEmbeddings {
  return new OpenAIEmbeddings({
    apiKey: env.dashscopeApiKey,
    model: env.embeddingModel,
    configuration: {
      baseURL: env.dashscopeBaseUrl,
    },
  });
}

function asUtcIso(date: Date): string {
  return date.toISOString().replace(/\.\d{3}Z$/, "Z");
}

function startOfDayUtc(date: Date): string {
  return asUtcIso(
    new Date(
      Date.UTC(
        date.getUTCFullYear(),
        date.getUTCMonth(),
        date.getUTCDate(),
        0,
        0,
        0,
      ),
    ),
  );
}

function endOfDayUtc(date: Date): string {
  return asUtcIso(
    new Date(
      Date.UTC(
        date.getUTCFullYear(),
        date.getUTCMonth(),
        date.getUTCDate(),
        23,
        59,
        59,
      ),
    ),
  );
}

export function normalizeExplicitTimeRange(input: {
  startTime?: string | null;
  endTime?: string | null;
  timeHint?: string | null;
}): { startTime: string | null; endTime: string | null } {
  if (input.startTime || input.endTime) {
    return {
      startTime: input.startTime ?? null,
      endTime: input.endTime ?? null,
    };
  }

  const hint = (input.timeHint ?? "").trim().toLowerCase();
  const now = new Date();

  if (!hint) {
    return { startTime: null, endTime: null };
  }

  if (hint.includes("今天")) {
    return { startTime: startOfDayUtc(now), endTime: endOfDayUtc(now) };
  }
  if (hint.includes("昨天")) {
    const target = new Date(now);
    target.setUTCDate(target.getUTCDate() - 1);
    return { startTime: startOfDayUtc(target), endTime: endOfDayUtc(target) };
  }
  if (hint.includes("最近") || hint.includes("最新") || hint.includes("刚才")) {
    const start = new Date(now);
    start.setUTCDate(start.getUTCDate() - 7);
    return { startTime: asUtcIso(start), endTime: asUtcIso(now) };
  }
  if (hint.includes("上周")) {
    const day = now.getUTCDay() || 7;
    const start = new Date(now);
    start.setUTCDate(start.getUTCDate() - day - 6);
    const end = new Date(start);
    end.setUTCDate(start.getUTCDate() + 6);
    return { startTime: startOfDayUtc(start), endTime: endOfDayUtc(end) };
  }
  if (hint.includes("去年")) {
    const year = now.getUTCFullYear() - 1;
    return {
      startTime: asUtcIso(new Date(Date.UTC(year, 0, 1, 0, 0, 0))),
      endTime: asUtcIso(new Date(Date.UTC(year, 11, 31, 23, 59, 59))),
    };
  }

  return { startTime: null, endTime: null };
}

function safeArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string =>
      typeof item === "string" && item.trim().length > 0
    )
    : [];
}

function tokenize(text: string): string[] {
  return text.toLowerCase()
    .split(/[\s,，。；;、:：|/]+/)
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function escapeIlike(value: string): string {
  return value.replace(/[%_,]/g, " ").trim();
}

function computeKeywordScore(query: string, chunks: string[]): number {
  const tokens = tokenize(query);
  if (tokens.length === 0) return 0;

  const haystack = chunks.join(" ").toLowerCase();
  let hits = 0;
  for (const token of tokens) {
    if (haystack.includes(token)) {
      hits += 1;
    }
  }
  return hits / tokens.length;
}

function normalizeMatrix(input: unknown): number[] | number[][] | null {
  if (!Array.isArray(input)) return null;

  if (
    input.length === 16 &&
    input.every((value) => Number.isFinite(Number(value)))
  ) {
    return input.map((value) => Number(value));
  }

  if (
    input.length === 4 &&
    input.every((row) =>
      Array.isArray(row) && row.length === 4 &&
      row.every((value) => Number.isFinite(Number(value)))
    )
  ) {
    return input.map((row) => (row as unknown[]).map((value) => Number(value)));
  }

  return null;
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

function derivePosesPath(scene: SceneCandidate): string | null {
  if (!scene.userId || !scene.sceneId) return null;
  return `${scene.userId}/${scene.sceneId}/output/webgl_poses.json`;
}

function summarizeToolResult(toolName: string, count: number): string {
  return `${toolName} 返回 ${count} 条候选`;
}

function serializeAssetOperation(state: AssetToolState) {
  return state.operation
    ? {
      tool_name: state.operation.tool_name,
      dry_run: state.operation.dry_run,
      requires_confirmation: state.operation.requires_confirmation,
      affected_count: state.operation.affected_count,
      preview: state.operation.preview,
    }
    : null;
}

function serializeAssetContext(state: AssetToolState) {
  return {
    last_tool_name: state.lastToolName,
    list: state.list,
    bundle: state.bundle,
    comparison: state.comparison,
    operation: serializeAssetOperation(state),
    pose_summary: state.poseSummary,
    related_models: state.relatedModels,
    place_versions: state.placeVersions,
    collection_summary: state.collectionSummary,
    thread_grouping: state.threadGrouping,
  };
}

async function classifyAgentMode(
  model: ChatOpenAI,
  query: string,
  options: SpatialSearchAgentOptions = {},
): Promise<AgentMode> {
  const normalized = query.trim().toLowerCase();
  const currentMode = options.currentMode ?? null;

  if (isDirectReplyQuery(query)) {
    return "chat";
  }

  if (
    currentMode === "compare" ||
    /比较|对比|变化|前后|两个月前|现在/.test(query)
  ) {
    return "time_compare";
  }
  if (/导览|旁白|脚本|大纲|故事|创作|生成一个.*记忆集/.test(query)) {
    return "creative";
  }
  if (/越来越|趋势|缺失|时间线|最近三次|长期记忆|关系摘要/.test(query)) {
    return "memory_graph";
  }
  if (
    currentMode === "batch_edit" ||
    currentMode === "collection" ||
    /改名|重命名|批量|标签|描述|摘要|专题|归档|集合|collection|对比.*模型|模型.*对比/.test(query) ||
    ((options.selectedModelIds?.length ?? 0) > 0 &&
      /这些模型|这几个模型|选中的模型|这三个模型/.test(query))
  ) {
    return "asset_metadata";
  }
  if (
    normalized.includes("找") ||
    normalized.includes("在哪") ||
    normalized.includes("有没有") ||
    normalized.includes("空间") ||
    normalized.includes("场景")
  ) {
    return "spatial_search";
  }

  const { buildAgentContextBlock } = await import("./prompts/context.ts");
  const { getRoutePrompt } = await import("./prompts/route.ts");
  const contextBlock = buildAgentContextBlock(options);

  const structuredModel = model.withStructuredOutput(agentRouteSchema);
  const result = await structuredModel.invoke([
    new SystemMessage(getRoutePrompt(contextBlock)),
    new HumanMessage(query),
  ]);
  return result.mode;
}

export function isDirectReplyQuery(query: string): boolean {
  const normalized = query
    .trim()
    .toLowerCase()
    .replace(/[！!。.,，?？~～\s]+/g, "");

  if (!normalized) {
    return false;
  }

  return [
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "hello",
    "hi",
    "在吗",
    "在不在",
    "有人吗",
    "谢谢",
    "多谢",
    "谢了",
    "辛苦了",
  ].includes(normalized);
}

function buildDirectReplyAnswer(query: string): string {
  const normalized = query
    .trim()
    .toLowerCase()
    .replace(/[！!。.,，?？~～\s]+/g, "");

  if (["谢谢", "多谢", "谢了", "辛苦了"].includes(normalized)) {
    return "不客气，我在。你可以直接说要找什么场景、比较哪个时间段，或者想整理哪些模型。";
  }

  return "你好，我在。你可以直接告诉我想找的场景/物体、要比较的时间段，或者要整理的模型。";
}

async function parseSpatialIntent(
  model: ChatOpenAI,
  query: string,
  options: SpatialSearchAgentOptions = {}
): Promise<SpatialIntent> {
  const trimmed = query.trim();
  const heuristicTimeHint = (
    trimmed.match(/今天|昨天|最近|最新|刚才|上周|去年/g) ?? []
  ).join(" ");
  const heuristicTargetType: SearchTargetType =
    /最近|最新|今天|昨天|上周|去年/.test(trimmed)
      ? "time"
      : /角落|窗边|书桌|桌面|厨房|客厅|卧室|门口|沙发旁/.test(trimmed)
      ? "location"
      : /场景|空间|房间/.test(trimmed)
      ? "scene"
      : "object";

  const heuristic: SpatialIntent = {
    rewrittenQuery: trimmed
      .replace(/^(帮我|给我|请你|麻烦你|找一下|帮我找|请帮我找)/, "")
      .replace(/(在哪|在哪里|还在吗|有没有|给我看看|帮我找出来)$/g, "")
      .trim() || trimmed,
    targetType: heuristicTargetType,
    objectHint: heuristicTargetType === "object" ? trimmed : null,
    locationHint: trimmed.match(/角落|窗边|书桌|桌面|厨房|客厅|卧室|门口|沙发旁/)?.[0] ?? null,
    sceneHint: null,
    timeHint: heuristicTimeHint || null,
    startTime: null,
    endTime: null,
    reasoning: "优先使用规则解析，减少同步路径上的 LLM 延迟。",
  };
  const heuristicRange = normalizeExplicitTimeRange(heuristic);
  if (
    heuristic.rewrittenQuery &&
    (heuristic.targetType !== "scene" || heuristic.locationHint !== null || heuristic.timeHint !== null)
  ) {
    return {
      ...heuristic,
      startTime: heuristicRange.startTime,
      endTime: heuristicRange.endTime,
    };
  }

  const structuredModel = model.withStructuredOutput(spatialIntentSchema);
  const today = new Date().toISOString().slice(0, 10);
  
  const { buildAgentContextBlock } = await import("./prompts/context.ts");
  const { getSpatialIntentPrompt } = await import("./prompts/spatial_intent.ts");
  const contextBlock = buildAgentContextBlock(options);

  const parsed = await structuredModel.invoke([
    new SystemMessage(getSpatialIntentPrompt(today, contextBlock)),
    new HumanMessage(query),
  ]);

  const normalizedRange = normalizeExplicitTimeRange(parsed);
  return {
    ...parsed,
    startTime: normalizedRange.startTime,
    endTime: normalizedRange.endTime,
  };
}

async function buildPoseTool(
  supabase: SupabaseClient,
  embeddings: OpenAIEmbeddings,
): Promise<DynamicStructuredTool> {
  return new DynamicStructuredTool({
    name: "pose_semantic_search",
    description:
      "当用户要找具体物体、位置、镜头视角或热点区域时使用。会返回最相近的 scene 和 pose 候选。",
    schema: z.object({
      query: z.string().min(1),
      threshold: z.number().min(0).max(1).default(0.35),
      limit: z.number().int().min(1).max(10).default(5),
      startTime: z.string().datetime({ offset: true }).nullable().default(null),
      endTime: z.string().datetime({ offset: true }).nullable().default(null),
    }),
    func: async ({ query, threshold, limit, startTime, endTime }) => {
      const queryEmbedding = await embeddings.embedQuery(query);
      const { data, error } = await supabase.rpc("match_memory_poses", {
        query_embedding: queryEmbedding,
        match_threshold: threshold,
        match_count: limit,
        filter_start: startTime,
        filter_end: endTime,
      } as never) as { data: unknown; error: { message: string } | null };

      if (error) {
        throw new Error(`pose_semantic_search 执行失败: ${error.message}`);
      }

      const rows = Array.isArray(data)
        ? data as Array<Record<string, unknown>>
        : [];
      const enriched: PoseSearchRow[] = [];

      for (const row of rows) {
        const modelId = String(row.id ?? "");
        const rawFrames = Array.isArray(row.matched_frames)
          ? row.matched_frames as Array<Record<string, unknown>>
          : [];
        const imageNames = rawFrames
          .map((frame) =>
            typeof frame.image_name === "string" ? frame.image_name : ""
          )
          .filter((value) => value.length > 0);

        const tagMap = new Map<string, string | null>();
        if (modelId && imageNames.length > 0) {
          const { data: poseRows } = await supabase
            .from("memory_poses")
            .select("image_name, tag")
            .eq("model_id", modelId)
            .in("image_name", imageNames);

          for (const poseRow of poseRows ?? []) {
            if (typeof poseRow.image_name === "string") {
              tagMap.set(
                poseRow.image_name,
                typeof poseRow.tag === "string" ? poseRow.tag : null,
              );
            }
          }
        }

        enriched.push({
          id: modelId,
          scene_id: String(row.scene_id ?? ""),
          description: typeof row.description === "string"
            ? row.description
            : null,
          ply_path: typeof row.ply_path === "string" ? row.ply_path : null,
          created_at: String(row.created_at ?? ""),
          user_id: typeof row.user_id === "string" ? row.user_id : null,
          similarity: Number(row.similarity ?? 0),
          matched_frames: rawFrames.map((frame) => ({
            image_name: String(frame.image_name ?? ""),
            transform_matrix: normalizeMatrix(frame.transform_matrix),
            similarity: Number(frame.similarity ?? 0),
            tag: tagMap.get(String(frame.image_name ?? "")) ?? null,
          })),
        });
      }

      return JSON.stringify(enriched);
    },
  });
}

async function buildSceneTool(
  supabase: SupabaseClient,
): Promise<DynamicStructuredTool> {
  return new DynamicStructuredTool({
    name: "scene_metadata_search",
    description:
      "当用户要找整个场景、时间范围内的记录，或需要按 scene/描述/标签筛选时使用。",
    schema: z.object({
      query: z.string().default(""),
      sceneId: z.string().nullable().default(null),
      limit: z.number().int().min(1).max(20).default(8),
      startTime: z.string().datetime({ offset: true }).nullable().default(null),
      endTime: z.string().datetime({ offset: true }).nullable().default(null),
    }),
    func: async ({ query, sceneId, limit, startTime, endTime }) => {
      let builder = supabase
        .from("model_assets")
        .select(
          "id, scene_id, user_id, description, objects, tags, ply_path, preview_img_path, meta_info, created_at",
        )
        .order("created_at", { ascending: false })
        .limit(Math.max(limit * 8, 40));

      if (sceneId) {
        builder = builder.eq("scene_id", sceneId);
      }
      if (startTime) {
        builder = builder.gte("created_at", startTime);
      }
      if (endTime) {
        builder = builder.lte("created_at", endTime);
      }
      if (query.trim()) {
        const keyword = escapeIlike(query);
        if (keyword) {
          builder = builder.or(
            `scene_id.ilike.%${keyword}%,description.ilike.%${keyword}%`,
          );
        }
      }

      const { data, error } = await builder;
      if (error) {
        throw new Error(`scene_metadata_search 执行失败: ${error.message}`);
      }

      const rows = (data ?? []) as SceneRow[];
      const scored = rows
        .map((row) => {
          const chunks = [
            row.scene_id,
            row.description ?? "",
            ...safeArray(row.objects),
            ...safeArray(row.tags),
          ];
          const keywordScore = query.trim()
            ? computeKeywordScore(query, chunks)
            : 0.45;
          return {
            ...row,
            keyword_score: keywordScore,
          };
        })
        .sort((a, b) => b.keyword_score - a.keyword_score)
        .slice(0, limit);

      return JSON.stringify(scored);
    },
  });
}

async function buildRecentSceneTool(
  supabase: SupabaseClient,
): Promise<DynamicStructuredTool> {
  return new DynamicStructuredTool({
    name: "recent_scene_search",
    description: "当用户主要按时间找最近或某段时间内的内容时使用。",
    schema: z.object({
      limit: z.number().int().min(1).max(10).default(5),
      startTime: z.string().datetime({ offset: true }).nullable().default(null),
      endTime: z.string().datetime({ offset: true }).nullable().default(null),
    }),
    func: async ({ limit, startTime, endTime }) => {
      let builder = supabase
        .from("model_assets")
        .select(
          "id, scene_id, user_id, description, objects, tags, ply_path, preview_img_path, meta_info, created_at",
        )
        .order("created_at", { ascending: false })
        .limit(limit);

      if (startTime) {
        builder = builder.gte("created_at", startTime);
      }
      if (endTime) {
        builder = builder.lte("created_at", endTime);
      }

      const { data, error } = await builder;
      if (error) {
        throw new Error(`recent_scene_search 执行失败: ${error.message}`);
      }

      return JSON.stringify((data ?? []) as SceneRow[]);
    },
  });
}

function mergeSceneCandidate(
  candidates: Map<string, SceneCandidate>,
  partial: SceneCandidate,
): void {
  const existing = candidates.get(partial.sceneId);
  if (!existing) {
    candidates.set(partial.sceneId, partial);
    return;
  }

  existing.sourceScores = { ...existing.sourceScores, ...partial.sourceScores };
  if (!existing.description && partial.description) {
    existing.description = partial.description;
  }
  if (!existing.plyPath && partial.plyPath) existing.plyPath = partial.plyPath;
  if (!existing.previewImgPath && partial.previewImgPath) {
    existing.previewImgPath = partial.previewImgPath;
  }
  if (!existing.bestPose && partial.bestPose) {
    existing.bestPose = partial.bestPose;
  }
  existing.objects = [...new Set([...existing.objects, ...partial.objects])];
  existing.tags = [...new Set([...existing.tags, ...partial.tags])];
}

function collectSceneCandidates(
  toolName: string,
  payload: string,
  candidates: Map<string, SceneCandidate>,
): number {
  const rows = JSON.parse(payload) as Array<Record<string, unknown>>;
  if (!Array.isArray(rows)) return 0;

  for (const row of rows) {
    if (toolName === "pose_semantic_search") {
      const frames = Array.isArray(row.matched_frames)
        ? row.matched_frames as Array<Record<string, unknown>>
        : [];
      const sortedFrames = frames
        .map((frame) => ({
          image_name: String(frame.image_name ?? ""),
          transform_matrix: normalizeMatrix(frame.transform_matrix),
          similarity: Number(frame.similarity ?? 0),
          tag: typeof frame.tag === "string" ? frame.tag : null,
        }))
        .sort((a, b) => b.similarity - a.similarity);
      mergeSceneCandidate(candidates, {
        modelId: String(row.id ?? ""),
        sceneId: String(row.scene_id ?? ""),
        userId: typeof row.user_id === "string" ? row.user_id : null,
        description: typeof row.description === "string" ? row.description : "",
        objects: [],
        tags: sortedFrames.map((frame) => frame.tag ?? "").filter((value) =>
          value.length > 0
        ),
        plyPath: typeof row.ply_path === "string" ? row.ply_path : null,
        previewImgPath: null,
        createdAt: String(row.created_at ?? ""),
        metaInfo: {},
        sourceScores: {
          pose_semantic_search: Number(row.similarity ?? 0),
        },
        bestPose: sortedFrames[0] ?? null,
      });
      continue;
    }

    mergeSceneCandidate(candidates, {
      modelId: String(row.id ?? ""),
      sceneId: String(row.scene_id ?? ""),
      userId: typeof row.user_id === "string" ? row.user_id : null,
      description: typeof row.description === "string" ? row.description : "",
      objects: safeArray(row.objects),
      tags: safeArray(row.tags),
      plyPath: typeof row.ply_path === "string" ? row.ply_path : null,
      previewImgPath: typeof row.preview_img_path === "string"
        ? row.preview_img_path
        : null,
      createdAt: String(row.created_at ?? ""),
      metaInfo: row.meta_info && typeof row.meta_info === "object"
        ? row.meta_info as Record<string, unknown>
        : {},
      sourceScores: {
        [toolName]: Number(
          toolName === "recent_scene_search" ? 0.7 : row.keyword_score ?? 0.5,
        ),
      },
      bestPose: null,
    });
  }

  return rows.length;
}

export function scoreSceneCandidate(
  candidate: SceneCandidate,
  intent: Pick<SpatialIntent, "rewrittenQuery" | "targetType">,
): number {
  const poseScore = candidate.sourceScores.pose_semantic_search ?? 0;
  const sceneScore = candidate.sourceScores.scene_metadata_search ?? 0;
  const timeScore = candidate.sourceScores.recent_scene_search ?? 0;
  const lexicalScore = computeKeywordScore(intent.rewrittenQuery, [
    candidate.sceneId,
    candidate.description,
    ...candidate.objects,
    ...candidate.tags,
    candidate.bestPose?.tag ?? "",
  ]);

  const weights: Record<SearchTargetType, [number, number, number, number]> = {
    object: [0.46, 0.2, 0.12, 0.22],
    location: [0.42, 0.18, 0.14, 0.26],
    time: [0.18, 0.18, 0.42, 0.22],
    scene: [0.2, 0.42, 0.14, 0.24],
  };
  const [poseWeight, sceneWeight, timeWeight, lexicalWeight] =
    weights[intent.targetType];

  return clamp(
    poseScore * poseWeight +
      sceneScore * sceneWeight +
      timeScore * timeWeight +
      lexicalScore * lexicalWeight,
    0,
    1,
  );
}

export function buildVisualizationActions(input: {
  scene: SceneCandidate | null;
  selectedPose: PoseFrame | null;
  supabase: SupabaseClient;
  bucket: string;
}): VisualizationAction[] {
  const { scene, selectedPose, supabase, bucket } = input;
  if (!scene) return [];

  const plyUrl = publicUrlForPath(supabase, bucket, scene.plyPath);
  const posesPath = derivePosesPath(scene);
  const posesUrl = publicUrlForPath(supabase, bucket, posesPath);
  const matrix = selectedPose?.transform_matrix ?? null;
  const imageId = selectedPose?.image_name ?? null;
  const hotspotLabel = selectedPose?.tag ?? scene.description ?? scene.sceneId;

  const actions: VisualizationAction[] = [
    {
      type: "open_scene",
      title: `打开场景 ${scene.sceneId}`,
      payload: {
        sceneId: scene.sceneId,
        modelId: scene.modelId,
        ply: plyUrl,
        poses: posesUrl,
      },
    },
  ];

  if (matrix) {
    actions.push({
      type: "fly_to_pose",
      title: imageId ? `飞到视角 ${imageId}` : "飞到最佳视角",
      payload: {
        sceneId: scene.sceneId,
        imageId,
        matrix,
      },
    });

  }

  return actions;
}

function getPreferredToolOrder(intent: SpatialIntent): string[] {
  switch (intent.targetType) {
    case "object":
    case "location":
      return ["pose_semantic_search", "scene_metadata_search"];
    case "time":
      return ["recent_scene_search", "scene_metadata_search"];
    case "scene":
      return ["scene_metadata_search", "recent_scene_search"];
    default:
      return ["scene_metadata_search"];
  }
}

function buildToolArgs(
  toolName: string,
  intent: SpatialIntent,
): Record<string, unknown> {
  if (toolName === "pose_semantic_search") {
    return {
      query: intent.rewrittenQuery,
      threshold: 0.35,
      limit: 5,
      startTime: intent.startTime,
      endTime: intent.endTime,
    };
  }

  if (toolName === "scene_metadata_search") {
    return {
      query: intent.rewrittenQuery,
      sceneId: intent.sceneHint,
      limit: 8,
      startTime: intent.startTime,
      endTime: intent.endTime,
    };
  }

  return {
    limit: 5,
    startTime: intent.startTime,
    endTime: intent.endTime,
  };
}

export function summarizeCandidateEvidence(
  candidates: Map<string, SceneCandidate>,
  intent: Pick<SpatialIntent, "rewrittenQuery" | "targetType">,
): {
  candidateCount: number;
  topScore: number;
  hasMultiSourceEvidence: boolean;
} {
  const ranked = [...candidates.values()]
    .map((candidate) => scoreSceneCandidate(candidate, intent))
    .sort((a, b) => b - a);

  const hasMultiSourceEvidence = [...candidates.values()].some((candidate) =>
    Object.keys(candidate.sourceScores).length >= 2
  );

  return {
    candidateCount: candidates.size,
    topScore: ranked[0] ?? 0,
    hasMultiSourceEvidence,
  };
}

export function shouldForceAnotherToolRound(input: {
  intent: SpatialIntent;
  candidates: Map<string, SceneCandidate>;
  trace: ToolTraceEntry[];
}): boolean {
  const { intent, candidates, trace } = input;
  const evidence = summarizeCandidateEvidence(candidates, intent);
  const usedTools = new Set(trace.map((entry) => entry.toolName));
  const preferredTools = getPreferredToolOrder(intent);
  const hasUnusedPreferredTool = preferredTools.some((toolName) =>
    !usedTools.has(toolName)
  );

  if (evidence.candidateCount < MIN_AGENT_CANDIDATES) {
    return true;
  }
  if (evidence.topScore < MIN_AGENT_TOP_SCORE) {
    return true;
  }
  if (!evidence.hasMultiSourceEvidence && hasUnusedPreferredTool) {
    return true;
  }

  return false;
}

function buildForcedToolCall(input: {
  intent: SpatialIntent;
  trace: ToolTraceEntry[];
}): ToolCallLike | null {
  const { intent, trace } = input;
  const usedTools = new Set(trace.map((entry) => entry.toolName));
  const preferredTools = getPreferredToolOrder(intent);
  const nextTool = preferredTools.find((toolName) => !usedTools.has(toolName));

  if (!nextTool) {
    return null;
  }

  return {
    id: `forced-${trace.length + 1}-${nextTool}`,
    name: nextTool,
    args: buildToolArgs(nextTool, intent),
  };
}

async function selectBestResult(
  model: ChatOpenAI,
  intent: SpatialIntent,
  rankedCandidates: Array<SceneCandidate & { score: number }>,
  actions: VisualizationAction[],
  options: SpatialSearchAgentOptions = {}
): Promise<SelectionResult> {
  const best = rankedCandidates[0] ?? null;
  if (best) {
    return {
      selectedSceneId: best.sceneId,
      selectedModelId: best.modelId,
      selectedPoseImageId: best.bestPose?.image_name ?? null,
      selectionReason: "综合候选得分、证据来源数量与关键词命中度后选择最高分结果",
      confidence: best.score,
      answer: `已找到最相关的空间结果：${best.description || best.sceneId}。证据来自${Object.keys(best.sourceScores).join(" + ")}。你可以直接打开场景，或查看其他候选。`,
      actions,
    };
  }

  const structuredModel = model.withStructuredOutput(selectionSchema);

  const { buildAgentContextBlock } = await import("./prompts/context.ts");
  const { getSelectionPrompt } = await import("./prompts/selection.ts");
  const contextBlock = buildAgentContextBlock(options);

  return await structuredModel.invoke([
    new SystemMessage(getSelectionPrompt(contextBlock)),
    new HumanMessage(JSON.stringify({
      intent,
      candidates: rankedCandidates.slice(0, 5).map((candidate) => ({
        sceneId: candidate.sceneId,
        modelId: candidate.modelId,
        score: candidate.score,
        description: candidate.description,
        tags: candidate.tags,
        bestPose: candidate.bestPose,
      })),
      suggestedActions: actions,
      defaultSelection: null,
    })),
  ]);
}

async function executeAgentToolLoop(input: {
  model: ChatOpenAI;
  intent: SpatialIntent;
  tools: DynamicStructuredTool[];
  options?: SpatialSearchAgentOptions;
}): Promise<
  { candidates: Map<string, SceneCandidate>; trace: ToolTraceEntry[] }
> {
  const { intent, tools } = input;
  const toolsByName = new Map(tools.map((tool) => [tool.name, tool]));
  const candidates = new Map<string, SceneCandidate>();
  const trace: ToolTraceEntry[] = [];
  const preferredTools = getPreferredToolOrder(intent);

  for (let round = 0; round < Math.min(MAX_AGENT_TOOL_ROUNDS, preferredTools.length); round += 1) {
    const toolName = preferredTools[round];
    const tool = toolsByName.get(toolName);
    if (!tool) {
      continue;
    }
    const args = buildToolArgs(toolName, intent);
    const toolResult = await tool.invoke(args);
    const resultText = typeof toolResult === "string"
      ? toolResult
      : JSON.stringify(toolResult);
    const count = collectSceneCandidates(tool.name, resultText, candidates);
    trace.push({
      toolName: tool.name,
      args,
      resultSummary: summarizeToolResult(tool.name, count),
    });

    if (
      !shouldForceAnotherToolRound({
        intent,
        candidates,
        trace,
      })
    ) {
      break;
    }
  }

  return { candidates, trace };
}

async function executeAssetToolLoop(input: {
  model: ChatOpenAI;
  query: string;
  tools: DynamicStructuredTool[];
  options?: SpatialSearchAgentOptions;
}): Promise<{ trace: ToolTraceEntry[]; state: AssetToolState }> {
  const { model, query, tools, options = {} } = input;
  const toolsByName = new Map(tools.map((tool) => [tool.name, tool]));
  const trace: ToolTraceEntry[] = [];
  const state = createEmptyAssetToolState();
  const agentModel = model.bindTools(tools);
  const today = new Date().toISOString().slice(0, 10);
  
  const { buildAgentContextBlock } = await import("./prompts/context.ts");
  const { getAssetToolLoopPrompt } = await import("./prompts/asset_tool_loop.ts");
  const contextBlock = buildAgentContextBlock(options);

  const messages = [
    new SystemMessage(getAssetToolLoopPrompt(today, contextBlock)),
    new HumanMessage(query),
  ];

  for (let round = 0; round < MAX_AGENT_TOOL_ROUNDS; round += 1) {
    const response = await agentModel.invoke(messages);
    messages.push(response);

    const toolCalls = Array.isArray(response.tool_calls)
      ? response.tool_calls
      : [];
    if (toolCalls.length === 0) {
      break;
    }

    for (const toolCall of toolCalls) {
      const tool = toolsByName.get(toolCall.name);
      if (!tool) {
        messages.push(
          new ToolMessage({
            tool_call_id: toolCall.id ?? toolCall.name,
            content: JSON.stringify({ error: `未知工具 ${toolCall.name}` }),
          }),
        );
        continue;
      }

      const toolResult = await tool.invoke(toolCall.args);
      const resultText = typeof toolResult === "string"
        ? toolResult
        : JSON.stringify(toolResult);
      const count = collectAssetToolResult(tool.name, resultText, state);
      trace.push({
        toolName: tool.name,
        args: toolCall.args ?? {},
        resultSummary: summarizeToolResult(tool.name, count),
      });
      messages.push(
        new ToolMessage({
          tool_call_id: toolCall.id ?? toolCall.name,
          content: resultText,
        }),
      );
    }
  }

  return { trace, state };
}

function emptyViewerPayload() {
  return {
    ply: null,
    poses: null,
    matrix: null,
    imageId: null,
  };
}

async function buildTimeCompareModeResponse(
  query: string,
  options: SpatialSearchAgentOptions,
): Promise<SpatialSearchResponse> {
  const result = await runTimeCompareAgent(query);
  const selectedReason = result.comparison.target
    ? "优先选择最近时间窗口中的目标版本作为主候选"
    : "当前仅命中单侧时间窗口，保留现有可信证据";

  const topCandidates = [result.comparison.target, result.comparison.baseline]
    .filter((item): item is NonNullable<typeof item> => Boolean(item))
    .map((item) => ({
      scene_id: item.sceneId,
      model_id: item.modelId,
      score: item.similarity,
      description: item.description ?? item.displayName ?? item.sceneId,
      pose_image_id: item.matchedFrames[0]?.imageName ?? null,
    }));

  return {
    success: true,
    mode: "time_compare",
    intent: null,
    selection: {
      scene_id: result.comparison.target?.sceneId ?? result.comparison.baseline?.sceneId ?? null,
      model_id: result.comparison.target?.modelId ?? result.comparison.baseline?.modelId ?? null,
      pose_image_id: result.comparison.target?.matchedFrames[0]?.imageName ??
        result.comparison.baseline?.matchedFrames[0]?.imageName ??
        null,
      confidence: result.comparison.target?.similarity ?? result.comparison.baseline?.similarity ?? 0,
      reason: selectedReason,
    },
    answer: result.answer,
    actions: result.actions.map((action) => ({
      type: action.type,
      title: action.type === "open_scene"
        ? `打开${action.slot === "baseline" ? "旧版" : "新版"}场景`
        : `飞到${action.slot === "baseline" ? "旧版" : "新版"}视角`,
      payload: { ...action },
    })),
    viewer_payload: emptyViewerPayload(),
    evidence: {
      baseline: result.comparison.baseline,
      target: result.comparison.target,
      diff: result.comparison.diff,
    },
    candidates: topCandidates,
    top_candidates: topCandidates,
    selected_candidate_reason: selectedReason,
    tool_trace: result.toolTrace.map((entry) => ({
      toolName: entry.toolName,
      args: entry.args,
      resultSummary: entry.resultSummary,
    })),
    asset_context: serializeAssetContext(createEmptyAssetToolState()),
    compare_context: {
      baseline: result.comparison.baseline,
      target: result.comparison.target,
      diff: result.comparison.diff,
      windows: result.intent,
    },
    collection_context: null,
    creative_context: null,
    memory_graph_context: null,
  };
}

async function buildCreativeModeResponse(input: {
  supabase: SupabaseClient;
  query: string;
  options: SpatialSearchAgentOptions;
}): Promise<SpatialSearchResponse> {
  const modelIds = input.options.selectedModelIds ??
    (input.options.currentModelId ? [input.options.currentModelId] : []);
  if (modelIds.length === 0) {
    throw new Error("创作模式需要至少一个已选模型或当前模型上下文");
  }

  const storyContext = await prepareStoryContext(input.supabase, {
    modelIds,
  }, {
    selectedModelIds: modelIds,
  });
  const outline = generateStoryOutlineFromContext(storyContext, input.query);
  const task = input.options.executionMode === "execute"
    ? await enqueueCreativeTask(input.supabase, {
      query: input.query,
      modelIds,
      outline,
      currentSceneId: input.options.currentSceneId,
    })
    : null;

  return {
    success: true,
    mode: "creative",
    intent: null,
    selection: {
      scene_id: input.options.currentSceneId ?? null,
      model_id: modelIds[0] ?? null,
      pose_image_id: null,
      confidence: 0.78,
      reason: "已根据选中模型整理创作上下文并生成导览大纲",
    },
    answer: task
      ? `已创建创作任务，任务 ID 为 ${task.task_id}。你可以基于当前大纲继续生成正式旁白或视频脚本。`
      : `已生成 ${outline.outline.length} 段创作大纲。当前为预览模式，如需正式生成任务，请切换到 execute。`,
    actions: [],
    viewer_payload: emptyViewerPayload(),
    evidence: {
      model_count: storyContext.model_count,
      timeline_summary: storyContext.timeline_summary,
    },
    candidates: [],
    top_candidates: [],
    selected_candidate_reason: "创作模式不返回空间候选，而是返回素材上下文和大纲",
    tool_trace: [
      {
        toolName: "prepare_story_context",
        args: { modelIds },
        resultSummary: `prepare_story_context 读取 ${storyContext.model_count} 个模型`,
      },
      {
        toolName: "generate_story_outline",
        args: { query: input.query },
        resultSummary: `generate_story_outline 生成 ${outline.outline.length} 段大纲`,
      },
      ...(task
        ? [{
          toolName: "enqueue_creative_task",
          args: { executionMode: input.options.executionMode },
          resultSummary: `enqueue_creative_task 创建任务 ${task.task_id}`,
        }]
        : []),
    ],
    asset_context: serializeAssetContext(createEmptyAssetToolState()),
    compare_context: null,
    collection_context: {
      story_context: storyContext,
    },
    creative_context: {
      story_context: storyContext,
      outline,
      task,
    },
    memory_graph_context: null,
  };
}

async function buildMemoryGraphModeResponse(input: {
  supabase: SupabaseClient;
  query: string;
  options: SpatialSearchAgentOptions;
}): Promise<SpatialSearchResponse> {
  const focusModelId = input.options.currentModelId ?? input.options.selectedModelIds?.[0];
  if (!focusModelId) {
    throw new Error("长期记忆模式需要当前模型或至少一个已选模型");
  }

  const trend = await getRecentPlaceTrend(input.supabase, { modelId: focusModelId, lookback: 5 });
  const missing = await findMissingObjectPattern(input.supabase, {
    modelId: focusModelId,
    objectName: "耳机",
    lookback: 5,
  });
  const timeline = await summarizePlaceChangeTimeline(input.supabase, {
    modelId: focusModelId,
    limit: 8,
  });
  const graph = await buildPersonalMemoryGraphSummary(input.supabase, {
    modelId: focusModelId,
  });

  return {
    success: true,
    mode: "memory_graph",
    intent: null,
    selection: {
      scene_id: input.options.currentSceneId ?? null,
      model_id: focusModelId,
      pose_image_id: null,
      confidence: 0.72,
      reason: "已围绕当前模型生成地点趋势、缺失模式与关系摘要",
    },
    answer: `${trend.summary}${missing.missing ? ` ${missing.summary}` : ""} ${timeline.summary}`,
    actions: [],
    viewer_payload: emptyViewerPayload(),
    evidence: {
      trend,
      missing,
      timeline,
      graph,
    },
    candidates: [],
    top_candidates: [],
    selected_candidate_reason: "长期记忆模式以关系摘要为主，不输出空间候选列表",
    tool_trace: [
      {
        toolName: "get_recent_place_trend",
        args: { modelId: focusModelId },
        resultSummary: trend.summary,
      },
      {
        toolName: "find_missing_object_pattern",
        args: { modelId: focusModelId, objectName: "耳机" },
        resultSummary: missing.summary,
      },
      {
        toolName: "summarize_place_change_timeline",
        args: { modelId: focusModelId },
        resultSummary: timeline.summary,
      },
      {
        toolName: "build_personal_memory_graph_summary",
        args: { modelId: focusModelId },
        resultSummary: graph.summary,
      },
    ],
    asset_context: serializeAssetContext(createEmptyAssetToolState()),
    compare_context: null,
    collection_context: null,
    creative_context: null,
    memory_graph_context: {
      trend,
      missing,
      timeline,
      graph,
    },
  };
}

export async function runSpatialSearchAgent(
  query: string,
  options: SpatialSearchAgentOptions = {},
): Promise<SpatialSearchResponse> {
  const env = ensureRuntimeEnv();
  const supabase = createSupabaseAdminClient(env);
  const model = createChatModel(env);
  const mode = await classifyAgentMode(model, query, options);

  if (mode === "chat") {
    return {
      success: true,
      mode: "chat",
      intent: null,
      selection: {
        scene_id: null,
        model_id: null,
        pose_image_id: null,
        confidence: 1,
        reason: "该请求属于纯问候或致谢，不需要进入工具调用链路。",
      },
      answer: buildDirectReplyAnswer(query),
      actions: [],
      viewer_payload: emptyViewerPayload(),
      evidence: null,
      candidates: [],
      top_candidates: [],
      selected_candidate_reason: null,
      tool_trace: [],
      asset_context: serializeAssetContext(createEmptyAssetToolState()),
      compare_context: null,
      collection_context: null,
      creative_context: null,
      memory_graph_context: null,
    };
  }

  if (mode === "time_compare") {
    return await buildTimeCompareModeResponse(query, options);
  }

  if (mode === "creative") {
    return await buildCreativeModeResponse({
      supabase,
      query,
      options,
    });
  }

  if (mode === "memory_graph") {
    return await buildMemoryGraphModeResponse({
      supabase,
      query,
      options,
    });
  }

  if (mode === "asset_metadata") {
    const assetTools = [
      buildListModelAssetsTool(supabase, {
        selectedModelIds: options.selectedModelIds,
      }),
      buildRenameModelAssetTool(supabase, {
        selectedModelIds: options.selectedModelIds,
        allowWrite: options.executionMode === "execute",
      }),
      buildBatchPatchModelMetadataTool(supabase, {
        selectedModelIds: options.selectedModelIds,
        allowWrite: options.executionMode === "execute",
      }),
      buildGetModelAssetBundleTool(supabase, {
        selectedModelIds: options.selectedModelIds,
      }),
      buildCompareModelAssetsTool(supabase, {
        selectedModelIds: options.selectedModelIds,
      }),
      buildGetPoseSummaryTool(supabase, {
        selectedModelIds: options.selectedModelIds,
      }),
      buildFindRelatedModelsTool(supabase, {
        selectedModelIds: options.selectedModelIds,
      }),
      buildListPlaceVersionsTool(supabase),
      buildCreateMemoryCollectionTool(supabase, {
        selectedModelIds: options.selectedModelIds,
      }),
      buildAddModelsToCollectionTool(supabase, {
        selectedModelIds: options.selectedModelIds,
      }),
      buildSummarizeCollectionTool(supabase),
      buildGroupModelsIntoThreadTool(supabase, {
        selectedModelIds: options.selectedModelIds,
      }),
    ];
    const { trace, state } = await executeAssetToolLoop({
      model,
      query,
      tools: assetTools,
      options,
    });

    return {
      success: true,
      mode: "asset_metadata",
      intent: null,
      selection: {
        scene_id: null,
        model_id: null,
        pose_image_id: null,
        confidence: 0,
        reason: "当前请求属于模型资产元数据操作",
      },
      answer: buildAssetAnswer(state) ?? "当前没有生成有效的模型资产结果。",
      actions: [],
      viewer_payload: emptyViewerPayload(),
      evidence: state.poseSummary
        ? {
          pose_summary: state.poseSummary,
        }
        : state.relatedModels
        ? {
          related_models: state.relatedModels,
        }
        : null,
      candidates: [],
      top_candidates: [],
      selected_candidate_reason: state.lastToolName
        ? `资产模式最后一次有效工具为 ${state.lastToolName}`
        : null,
      tool_trace: trace,
      asset_context: serializeAssetContext(state),
      compare_context: state.placeVersions
        ? {
          place_versions: state.placeVersions,
        }
        : null,
      collection_context: state.collectionSummary
        ? {
          collection_summary: state.collectionSummary,
        }
        : null,
      creative_context: null,
      memory_graph_context: null,
    };
  }

  const embeddings = createEmbeddingsModel(env);

  const intent = await parseSpatialIntent(model, query, options);
  const tools = [
    await buildPoseTool(supabase, embeddings),
    await buildSceneTool(supabase),
    await buildRecentSceneTool(supabase),
  ];
  const { candidates: candidateMap, trace } = await executeAgentToolLoop({
    model,
    intent,
    tools,
    options,
  });

  const rankedCandidates = [...candidateMap.values()]
    .map((candidate) => ({
      ...candidate,
      score: scoreSceneCandidate(candidate, intent),
    }))
    .sort((a, b) => b.score - a.score);

  const bestCandidate = rankedCandidates[0] ?? null;
  const selectedPose = bestCandidate?.bestPose ?? null;
  const suggestedActions = buildVisualizationActions({
    scene: bestCandidate,
    selectedPose,
    supabase,
    bucket: env.storageBucket,
  });

  const selection = rankedCandidates.length > 0
    ? await selectBestResult(model, intent, rankedCandidates, suggestedActions, options)
    : {
      selectedSceneId: null,
      selectedModelId: null,
      selectedPoseImageId: null,
      selectionReason: "没有检索到可信候选",
      confidence: 0,
      answer: "当前没有找到可信的空间检索结果。",
      actions: [],
    };

  const finalScene =
    rankedCandidates.find((candidate) =>
      candidate.sceneId === selection.selectedSceneId
    ) ?? bestCandidate;
  const finalPose =
    finalScene?.bestPose?.image_name === selection.selectedPoseImageId
      ? finalScene.bestPose
      : finalScene?.bestPose ?? null;
  const finalActions = buildVisualizationActions({
    scene: finalScene ?? null,
    selectedPose: finalPose,
    supabase,
    bucket: env.storageBucket,
  });

  return {
    success: true,
    mode: "spatial_search",
    intent,
    selection: {
      scene_id: selection.selectedSceneId,
      model_id: selection.selectedModelId,
      pose_image_id: selection.selectedPoseImageId,
      confidence: selection.confidence,
      reason: selection.selectionReason,
    },
    answer: selection.answer,
    actions: finalActions,
    viewer_payload: {
      ply: finalScene
        ? publicUrlForPath(supabase, env.storageBucket, finalScene.plyPath)
        : null,
      poses: finalScene
        ? publicUrlForPath(
          supabase,
          env.storageBucket,
          derivePosesPath(finalScene),
        )
        : null,
      matrix: finalPose?.transform_matrix ?? null,
      imageId: finalPose?.image_name ?? null,
    },
    evidence: finalScene
      ? {
        sceneId: finalScene.sceneId,
        modelId: finalScene.modelId,
        similarity: selection.confidence,
        matchedFrames: finalPose
          ? [{
            imageName: finalPose.image_name,
            similarity: finalPose.similarity,
            transformMatrix: finalPose.transform_matrix,
            tag: finalPose.tag,
          }]
          : [],
        description: finalScene.description,
        tags: finalScene.tags,
      }
      : null,
    candidates: rankedCandidates.slice(0, 5).map((candidate) => ({
      scene_id: candidate.sceneId,
      model_id: candidate.modelId,
      score: candidate.score,
      description: candidate.description,
      pose_image_id: candidate.bestPose?.image_name ?? null,
    })),
    top_candidates: rankedCandidates.slice(0, 5).map((candidate) => ({
      scene_id: candidate.sceneId,
      model_id: candidate.modelId,
      score: candidate.score,
      description: candidate.description,
      pose_image_id: candidate.bestPose?.image_name ?? null,
    })),
    selected_candidate_reason: selection.selectionReason,
    tool_trace: trace,
    asset_context: serializeAssetContext(createEmptyAssetToolState()),
    compare_context: null,
    collection_context: null,
    creative_context: null,
    memory_graph_context: null,
  };
}
