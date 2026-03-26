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
  buildPersonalMemoryGraphSummary,
  buildSummarizeCollectionTool,
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
const DIRECT_REPLY_TOKENS = new Set([
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
]);

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

const flatMatrixSchema = z.array(z.number()).length(16);
const nestedMatrixSchema = z.array(z.array(z.number()).length(4)).length(4);
const matrixSchema = z.union([flatMatrixSchema, nestedMatrixSchema]);

const openSceneActionSchema = z.object({
  type: z.literal("open_scene"),
  title: z.string(),
  payload: z.object({
    sceneId: z.string(),
    modelId: z.string(),
    ply: z.string().nullable(),
    poses: z.string().nullable(),
  }),
});

const flyToPoseActionSchema = z.object({
  type: z.literal("fly_to_pose"),
  title: z.string(),
  payload: z.object({
    sceneId: z.string(),
    imageId: z.string().nullable(),
    matrix: matrixSchema.nullable(),
  }),
});

const visualizationActionSchema = z.discriminatedUnion("type", [
  openSceneActionSchema,
  flyToPoseActionSchema,
]);

const candidateSchema = z.object({
  scene_id: z.string(),
  model_id: z.string(),
  score: z.number().min(0).max(1),
  description: z.string(),
  pose_image_id: z.string().nullable(),
});

const toolTraceEntrySchema = z.object({
  toolName: z.string(),
  args: z.record(z.string(), z.unknown()),
  resultSummary: z.string(),
});

const selectionSummarySchema = z.object({
  scene_id: z.string().nullable(),
  model_id: z.string().nullable(),
  pose_image_id: z.string().nullable(),
  confidence: z.number().min(0).max(1),
  reason: z.string(),
});

const viewerPayloadSchema = z.object({
  ply: z.string().nullable(),
  poses: z.string().nullable(),
  matrix: matrixSchema.nullable(),
  imageId: z.string().nullable(),
});

const matchedFrameSchema = z.object({
  imageName: z.string(),
  similarity: z.number().min(0).max(1),
  transformMatrix: matrixSchema.nullable(),
  tag: z.string().nullable().optional(),
});

const spatialEvidenceSchema = z.object({
  sceneId: z.string(),
  modelId: z.string(),
  similarity: z.number().min(0).max(1),
  matchedFrames: z.array(matchedFrameSchema),
  description: z.string().nullable().optional(),
  tags: z.array(z.string()).optional(),
});

const poseSummarySchema = z.object({
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

const placeVersionsSchema = z.object({
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

const storyContextSchema = z.object({
  title: z.string(),
  model_count: z.number().int().min(0),
  ordered_models: z.array(z.object({
    model_id: z.string(),
    scene_id: z.string(),
    display_name: z.string().nullable(),
    summary_title: z.string().nullable(),
    version_label: z.string().nullable(),
    created_at: z.string(),
    tags: z.array(z.string()),
    objects: z.array(z.string()),
    description: z.string().nullable(),
  })),
  timeline_summary: z.string(),
  dominant_tags: z.array(z.string()),
});

const storyOutlineSchema = z.object({
  title: z.string(),
  outline: z.array(z.string()),
  narration_style: z.string(),
});

const creativeTaskSchema = z.object({
  task_id: z.string(),
}).passthrough().nullable();

const recentPlaceTrendSchema = z.object({
  place_id: z.string().nullable(),
  memory_thread_id: z.string().nullable(),
  related_models: z.array(z.string()),
  trend: z.string(),
  pose_counts: z.array(z.number()),
  object_counts: z.array(z.number()),
  summary: z.string(),
});

const missingObjectPatternSchema = z.object({
  object_name: z.string(),
  baseline_model_ids: z.array(z.string()),
  target_model_id: z.string().nullable(),
  missing: z.boolean(),
  summary: z.string(),
});

const placeTimelineSummarySchema = z.object({
  place_id: z.string().nullable(),
  memory_thread_id: z.string().nullable(),
  timeline: z.array(z.object({
    model_id: z.string(),
    created_at: z.string(),
    version_label: z.string().nullable(),
    display_name: z.string().nullable(),
  })),
  summary: z.string(),
});

const memoryGraphSummarySchema = z.object({
  focus_model_id: z.string(),
  related_model_ids: z.array(z.string()),
  place_id: z.string().nullable(),
  memory_thread_id: z.string().nullable(),
  summary: z.string(),
  key_relationships: z.array(z.string()),
});

const compareSceneEvidenceSchema = z.object({
  sceneId: z.string(),
  modelId: z.string(),
  displayName: z.string().nullable(),
  description: z.string().nullable(),
  createdAt: z.string(),
  similarity: z.number().min(0).max(1),
  objects: z.array(z.string()),
  tags: z.array(z.string()),
  matchedFrames: z.array(matchedFrameSchema),
  ply: z.string().nullable(),
  poses: z.string().nullable(),
}).nullable();

const timeCompareContextSchema = z.object({
  baseline: compareSceneEvidenceSchema,
  target: compareSceneEvidenceSchema,
  diff: z.object({
    commonObjects: z.array(z.string()),
    addedObjects: z.array(z.string()),
    removedObjects: z.array(z.string()),
    commonTags: z.array(z.string()),
    addedTags: z.array(z.string()),
    removedTags: z.array(z.string()),
    limitations: z.array(z.string()),
  }),
  windows: z.object({
    originalQuery: z.string(),
    parsedSearchText: z.string(),
    compareFocus: z.string().nullable(),
    baseline: z.object({
      startTime: z.string(),
      endTime: z.string(),
    }),
    target: z.object({
      startTime: z.string(),
      endTime: z.string(),
    }),
    reasoning: z.string(),
  }),
});

const collectionContextSchema = z.object({
  collection_summary: memoryCollectionSummarySchema,
}).nullable();

const creativeContextSchema = z.object({
  story_context: storyContextSchema,
  outline: storyOutlineSchema,
  task: creativeTaskSchema,
}).nullable();

const memoryGraphContextSchema = z.object({
  trend: recentPlaceTrendSchema,
  missing: missingObjectPatternSchema,
  timeline: placeTimelineSummarySchema,
  graph: memoryGraphSummarySchema,
}).nullable();

const candidateSceneRefSchema = z.object({
  index: z.number().int().min(1),
  sceneId: z.string(),
  modelId: z.string(),
  description: z.string(),
});

const sessionOperationPreviewSchema = z.object({
  toolName: z.string(),
  affectedCount: z.number().int().min(0),
  modelIds: z.array(z.string()).optional(),
  args: z.record(z.string(), z.unknown()).optional(),
}).nullable();

const sessionStateSchema = z.object({
  lastMode: agentModeSchema.optional(),
  lastSelectedModelIds: z.array(z.string()).optional(),
  lastCandidateRefs: z.array(candidateSceneRefSchema).optional(),
  lastOperationPreview: sessionOperationPreviewSchema.optional(),
});

const agentFollowUpSchema = z.object({
  status: z.enum(["idle", "waiting_user_input"]),
  kind: z.enum([
    "general",
    "rename_model",
    "choose_candidate",
    "confirm_write",
  ]),
  message: z.string(),
  input_placeholder: z.string().nullable().optional(),
  suggested_replies: z.array(z.string()).default([]),
}).nullable();

const assetContextSchema = z.object({
  last_tool_name: z.string().nullable(),
  list: z.array(z.object({
    id: z.string(),
    scene_id: z.string(),
    display_name: z.string().nullable(),
    description: z.string().nullable(),
    tags: z.array(z.string()),
    created_at: z.string(),
  })).nullable(),
  bundle: z.array(z.object({
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
  })).nullable(),
  comparison: z.object({
    rows: z.array(z.object({
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
    })),
    diff: z.object({
      common_tags: z.array(z.string()),
      common_objects: z.array(z.string()),
      tag_only_by_model: z.record(z.string(), z.array(z.string())),
      object_only_by_model: z.record(z.string(), z.array(z.string())),
      time_order: z.array(z.string()),
      pose_count_by_model: z.record(z.string(), z.number()),
    }),
  }).nullable(),
  operation: z.object({
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
  }).nullable(),
  pose_summary: poseSummarySchema.nullable().optional(),
  related_models: z.array(relatedModelSummarySchema).nullable().optional(),
  place_versions: placeVersionsSchema.nullable().optional(),
  collection_summary: memoryCollectionSummarySchema.nullable().optional(),
  thread_grouping: z.object({
    model_ids: z.array(z.string()),
    place_id: z.string(),
    memory_thread_id: z.string(),
  }).nullable().optional(),
});

const poseSearchRowSchema = z.object({
  id: z.string(),
  scene_id: z.string(),
  description: z.string().nullable(),
  ply_path: z.string().nullable(),
  created_at: z.string(),
  user_id: z.string().nullable(),
  similarity: z.number(),
  matched_frames: z.array(z.object({
    image_name: z.string(),
    transform_matrix: matrixSchema.nullable(),
    similarity: z.number(),
    tag: z.string().nullable(),
  })),
});

const sceneSearchRowSchema = z.object({
  id: z.string(),
  scene_id: z.string(),
  user_id: z.string().nullable(),
  description: z.string().nullable(),
  objects: z.array(z.string()).nullable(),
  tags: z.array(z.string()).nullable(),
  ply_path: z.string().nullable(),
  preview_img_path: z.string().nullable(),
  meta_info: z.record(z.string(), z.unknown()).nullable(),
  created_at: z.string(),
  keyword_score: z.number().optional(),
});

const poseSearchResultSchema = z.array(poseSearchRowSchema);
const sceneSearchResultSchema = z.array(sceneSearchRowSchema);

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
  sessionState?: z.infer<typeof sessionStateSchema> | null;
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

export type AgentProgressEvent =
  | {
    event: "status";
    data: {
      phase: string;
      summary: string;
      detail?: string;
    };
  }
  | {
    event: "tool_call";
    data: {
      name: string;
      args: Record<string, unknown>;
      summary?: string;
      round?: number;
    };
  }
  | {
    event: "tool_result";
    data: {
      name: string;
      summary: string;
      count?: number;
      round?: number;
    };
  };

type AgentRuntimeCallbacks = {
  onEvent?: (event: AgentProgressEvent) => void | Promise<void>;
};

const responseBaseSchema = z.object({
  success: z.literal(true),
  answer: z.string(),
  actions: z.array(visualizationActionSchema),
  selection: selectionSummarySchema,
  viewer_payload: viewerPayloadSchema,
  candidates: z.array(candidateSchema),
  top_candidates: z.array(candidateSchema),
  selected_candidate_reason: z.string().nullable(),
  tool_trace: z.array(toolTraceEntrySchema),
  asset_context: assetContextSchema,
  session_state: sessionStateSchema.nullable().optional(),
  conversation_summary: z.string().nullable().optional(),
  follow_up: agentFollowUpSchema.optional(),
});

const spatialSearchResponseSchema = responseBaseSchema.extend({
  mode: z.literal("spatial_search"),
  intent: spatialIntentSchema,
  evidence: spatialEvidenceSchema.nullable(),
  compare_context: z.null(),
  collection_context: z.null(),
  creative_context: z.null(),
  memory_graph_context: z.null(),
});

const assetMetadataResponseSchema = responseBaseSchema.extend({
  mode: z.literal("asset_metadata"),
  intent: z.null(),
  evidence: z.union([
    z.object({ pose_summary: poseSummarySchema }),
    z.object({ related_models: z.array(relatedModelSummarySchema) }),
    z.null(),
  ]),
  compare_context: z.object({
    place_versions: placeVersionsSchema,
  }).nullable(),
  collection_context: collectionContextSchema,
  creative_context: z.null(),
  memory_graph_context: z.null(),
});

const timeCompareResponseSchema = responseBaseSchema.extend({
  mode: z.literal("time_compare"),
  intent: z.null(),
  evidence: z.object({
    baseline: compareSceneEvidenceSchema,
    target: compareSceneEvidenceSchema,
    diff: timeCompareContextSchema.shape.diff,
  }),
  compare_context: timeCompareContextSchema,
  collection_context: z.null(),
  creative_context: z.null(),
  memory_graph_context: z.null(),
});

const creativeModeResponseSchema = responseBaseSchema.extend({
  mode: z.literal("creative"),
  intent: z.null(),
  evidence: z.object({
    model_count: z.number().int().min(0),
    timeline_summary: z.string(),
  }),
  compare_context: z.null(),
  collection_context: z.object({
    story_context: storyContextSchema,
  }),
  creative_context: creativeContextSchema,
  memory_graph_context: z.null(),
});

const memoryGraphModeResponseSchema = responseBaseSchema.extend({
  mode: z.literal("memory_graph"),
  intent: z.null(),
  evidence: memoryGraphContextSchema,
  compare_context: z.null(),
  collection_context: z.null(),
  creative_context: z.null(),
  memory_graph_context: memoryGraphContextSchema,
});

const spatialSearchResponseSchemaUnion = z.discriminatedUnion("mode", [
  spatialSearchResponseSchema,
  assetMetadataResponseSchema,
  timeCompareResponseSchema,
  creativeModeResponseSchema,
  memoryGraphModeResponseSchema,
]);

export type SessionState = z.infer<typeof sessionStateSchema>;
export type SpatialSearchResponse = z.infer<
  typeof spatialSearchResponseSchemaUnion
>;

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

export function shouldPreferHeuristicSpatialRoute(query: string): boolean {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return false;
  if (isDirectReplyQuery(normalized)) return false;
  if (
    /比较|对比|变化|前后|两个月前|现在|导览|旁白|脚本|大纲|故事|创作|趋势|缺失|时间线|长期记忆|关系摘要|改名|重命名|批量|标签|描述|摘要|专题|归档|集合|collection|模型.*对比/
      .test(normalized)
  ) {
    return false;
  }

  return /^(查一下|查下|看一下|看下|搜一下|搜下|搜一搜|查一查|帮我查一下|帮我看一下|请查一下|请看一下|看看|查查)/
    .test(normalized) ||
    /找|查|看|搜|检索|在哪|在哪里|有没有|空间|场景/.test(normalized);
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

async function emitProgress(
  callbacks: AgentRuntimeCallbacks | undefined,
  event: AgentProgressEvent,
): Promise<void> {
  if (callbacks?.onEvent) {
    await callbacks.onEvent(event);
    // 在 Deno 环境下，通过一个小延时强制触发事件循环，确保数据被写入 Stream
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

function normalizeDirectReplyQuery(query: string): string {
  return query
    .trim()
    .toLowerCase()
    .replace(/[！!。.,，?？~～\s]+/g, "");
}

export function isDirectReplyQuery(query: string): boolean {
  const normalized = normalizeDirectReplyQuery(query);
  if (!normalized) {
    return false;
  }
  return DIRECT_REPLY_TOKENS.has(normalized);
}

function buildDirectReplyAnswer(query: string): string {
  const normalized = normalizeDirectReplyQuery(query);

  if (["谢谢", "多谢", "谢了", "辛苦了"].includes(normalized)) {
    return "不客气，我在。你可以直接说要找什么场景、比较哪个时间段，或者想整理哪些模型。";
  }

  return "你好，我在。你可以直接告诉我想找的场景/物体、要比较的时间段，或者要整理的模型。";
}

async function classifyAgentMode(
  model: ChatOpenAI,
  query: string,
  options: SpatialSearchAgentOptions = {},
): Promise<AgentMode> {
  const normalized = query.trim().toLowerCase();
  const currentMode = options.currentMode ?? null;

  if (isDirectReplyQuery(query)) {
    return "spatial_search";
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
    /改名|重命名|批量|标签|描述|摘要|专题|归档|集合|collection|对比.*模型|模型.*对比/
      .test(query) ||
    ((options.selectedModelIds?.length ?? 0) > 0 &&
      /这些模型|这几个模型|选中的模型|这三个模型/.test(query))
  ) {
    return "asset_metadata";
  }
  if (
    shouldPreferHeuristicSpatialRoute(query) ||
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

type DeterministicAssetRenameIntent = {
  target:
    | { kind: "latest"; count: number }
    | { kind: "current"; modelId: string }
    | { kind: "selected"; modelIds: string[] }
    | { kind: "session"; modelIds: string[] };
  newName: string | null;
};

function parseChineseCountToken(token: string): number | null {
  const trimmed = token.trim();
  if (!trimmed) {
    return null;
  }

  if (/^\d+$/.test(trimmed)) {
    const value = Number(trimmed);
    return Number.isFinite(value) && value > 0 ? value : null;
  }

  const normalized = trimmed
    .replace(/两/g, "二")
    .replace(/俩/g, "二");
  const directMap: Record<string, number> = {
    一: 1,
    二: 2,
    三: 3,
    四: 4,
    五: 5,
    六: 6,
    七: 7,
    八: 8,
    九: 9,
    十: 10,
  };
  if (directMap[normalized] != null) {
    return directMap[normalized];
  }

  if (/^十[一二三四五六七八九]$/.test(normalized)) {
    return 10 + (directMap[normalized.slice(1)] ?? 0);
  }
  if (/^[一二三四五六七八九]十$/.test(normalized)) {
    return (directMap[normalized[0]!] ?? 0) * 10;
  }
  if (/^[一二三四五六七八九]十[一二三四五六七八九]$/.test(normalized)) {
    return (directMap[normalized[0]!] ?? 0) * 10 +
      (directMap[normalized[2]!] ?? 0);
  }

  return null;
}

function extractLatestModelCount(query: string): number {
  const patterns = [
    /最新(?:的)?([0-9一二三四五六七八九十两俩]+)(?:个|条)?模型/,
    /最近(?:的)?([0-9一二三四五六七八九十两俩]+)(?:个|条)?模型/,
    /([0-9一二三四五六七八九十两俩]+)(?:个|条)最新模型/,
    /([0-9一二三四五六七八九十两俩]+)(?:个|条)最近模型/,
  ];

  for (const pattern of patterns) {
    const matched = query.match(pattern)?.[1];
    const count = matched ? parseChineseCountToken(matched) : null;
    if (count) {
      return Math.max(1, Math.min(20, count));
    }
  }

  return 1;
}

function isConfirmWriteQuery(query: string): boolean {
  return /^(请)?(确认|执行|正式执行|继续执行)/.test(query.trim()) ||
    /确认执行|正式写入|开始执行|执行刚才|执行上一次|确认写入/.test(query);
}

function referencesMultipleModels(query: string): boolean {
  return /这些模型|这几个模型|这批模型|这三?个模型|这\d+个模型|它们|这几个/
    .test(
      query,
    );
}

function extractRenameTargetName(query: string): string | null {
  const trimmed = query.trim();
  const patterns = [
    /(?:改名为|重命名为|名字改成|名称改成|改成|叫做|命名为)\s*[“"「『]?([^”"」』。！？?,，\n]+)[”"」』]?/i,
    /(?:新名字|新名称)(?:是|叫)\s*[“"「『]?([^”"」』。！？?,，\n]+)[”"」』]?/i,
  ];

  for (const pattern of patterns) {
    const matched = trimmed.match(pattern)?.[1]?.trim();
    if (matched) {
      return matched.replace(/[。！？!,，]+$/g, "").trim() || null;
    }
  }

  return null;
}

export function parseDeterministicAssetRenameIntent(
  query: string,
  options: SpatialSearchAgentOptions = {},
): DeterministicAssetRenameIntent | null {
  const trimmed = query.trim();
  if (!trimmed) {
    return null;
  }

  const hasRenameIntent =
    /改名|重命名|修改.*名字|修改.*名称|模型名字|模型名称|名字改|名称改/.test(
      trimmed,
    );
  if (!hasRenameIntent) {
    return null;
  }

  const newName = extractRenameTargetName(trimmed);
  const latestCount = extractLatestModelCount(trimmed);

  if (/最新|最近|刚拍|刚刚|最后一个/.test(trimmed)) {
    return {
      target: { kind: "latest", count: latestCount },
      newName,
    };
  }

  if (
    options.currentModelId &&
    /当前模型|这个模型|当前这个模型|这个场景|当前场景/.test(trimmed)
  ) {
    return {
      target: { kind: "current", modelId: options.currentModelId },
      newName,
    };
  }

  if ((options.selectedModelIds?.length ?? 0) === 1) {
    return {
      target: {
        kind: "selected",
        modelIds: options.selectedModelIds!,
      },
      newName,
    };
  }

  if (
    (options.selectedModelIds?.length ?? 0) > 1 &&
    referencesMultipleModels(trimmed)
  ) {
    return {
      target: {
        kind: "selected",
        modelIds: options.selectedModelIds!,
      },
      newName,
    };
  }

  if (
    (options.sessionState?.lastSelectedModelIds?.length ?? 0) === 1 &&
    /它|这个|这个模型|该模型|刚才那个|上一轮那个|上一个模型/.test(trimmed)
  ) {
    return {
      target: {
        kind: "session",
        modelIds: options.sessionState!.lastSelectedModelIds!,
      },
      newName,
    };
  }

  if (
    (options.sessionState?.lastSelectedModelIds?.length ?? 0) > 1 &&
    referencesMultipleModels(trimmed)
  ) {
    return {
      target: {
        kind: "session",
        modelIds: options.sessionState!.lastSelectedModelIds!,
      },
      newName,
    };
  }

  return null;
}

async function replayPendingAssetWriteIfNeeded(input: {
  supabase: SupabaseClient;
  query: string;
  options: SpatialSearchAgentOptions;
  callbacks?: AgentRuntimeCallbacks;
}): Promise<SpatialSearchResponse | null> {
  const { query, options, supabase, callbacks } = input;
  if (!isConfirmWriteQuery(query)) {
    return null;
  }

  const pending = options.sessionState?.lastOperationPreview;
  if (!pending?.toolName || !pending.args) {
    return null;
  }

  if (options.executionMode !== "execute") {
    return finalizeResponse({
      success: true,
      mode: "asset_metadata",
      intent: null,
      selection: {
        scene_id: null,
        model_id: pending.modelIds?.[0] ?? null,
        pose_image_id: null,
        confidence: 0.82,
        reason: "检测到用户希望确认写入，但当前请求仍处于预览模式",
      },
      answer:
        "我已识别到你要确认执行，但当前请求还是 preview 模式。请切换到 execute 后再确认一次。",
      actions: [],
      viewer_payload: emptyViewerPayload(),
      evidence: null,
      candidates: [],
      top_candidates: [],
      selected_candidate_reason: "缺少 execute 模式，已阻止正式写入",
      tool_trace: [],
      asset_context: serializeAssetContext(createEmptyAssetToolState()),
      compare_context: null,
      collection_context: null,
      creative_context: null,
      memory_graph_context: null,
    });
  }

  const toolsByName = new Map<string, DynamicStructuredTool>([
    [
      "rename_model_asset",
      buildRenameModelAssetTool(supabase, {
        selectedModelIds: options.selectedModelIds,
        allowWrite: true,
      }),
    ],
    [
      "batch_patch_model_metadata",
      buildBatchPatchModelMetadataTool(supabase, {
        selectedModelIds: options.selectedModelIds,
        allowWrite: true,
      }),
    ],
  ]);
  const tool = toolsByName.get(pending.toolName);
  if (!tool) {
    return null;
  }

  const executionArgs = {
    ...pending.args,
    dryRun: false,
  };
  const state = createEmptyAssetToolState();
  const trace: ToolTraceEntry[] = [];

  await emitProgress(callbacks, {
    event: "status",
    data: {
      phase: "asset_write_replay",
      summary: "已识别到确认执行请求，准备重放上一轮预览操作",
    },
  });
  await emitProgress(callbacks, {
    event: "tool_call",
    data: {
      name: tool.name,
      args: executionArgs,
      summary: "开始正式执行上一轮已确认的资产写操作",
      round: 1,
    },
  });

  const toolResult = await tool.invoke(executionArgs);
  const resultText = typeof toolResult === "string"
    ? toolResult
    : JSON.stringify(toolResult);
  const count = collectAssetToolResult(tool.name, resultText, state);
  const resultSummary = summarizeToolResult(tool.name, count);
  trace.push({
    toolName: tool.name,
    args: executionArgs,
    resultSummary,
  });
  await emitProgress(callbacks, {
    event: "tool_result",
    data: {
      name: tool.name,
      summary: resultSummary,
      count,
      round: 1,
    },
  });

  const selectionModelId = state.operation?.preview[0]?.model_id ??
    pending.modelIds?.[0] ?? null;
  const selectionSceneId = state.operation?.preview[0]?.scene_id ?? null;

  return finalizeResponse({
    success: true,
    mode: "asset_metadata",
    intent: null,
    selection: {
      scene_id: selectionSceneId,
      model_id: selectionModelId,
      pose_image_id: null,
      confidence: 0.97,
      reason: "已按上一轮预览参数正式执行资产写操作",
    },
    answer: tool.name === "rename_model_asset"
      ? `已正式执行改名：${
        state.operation?.preview[0]?.old_display_name ??
          selectionSceneId ?? "该模型"
      } -> ${state.operation?.preview[0]?.new_display_name ?? "新名称"}。`
      : `已正式执行批量元数据修改，影响 ${
        state.operation?.affected_count ?? count
      } 个模型。`,
    actions: [],
    viewer_payload: emptyViewerPayload(),
    evidence: null,
    candidates: [],
    top_candidates: [],
    selected_candidate_reason: "已根据会话中的预览参数完成正式写入",
    tool_trace: trace,
    asset_context: serializeAssetContext(state),
    compare_context: null,
    collection_context: null,
    creative_context: null,
    memory_graph_context: null,
  });
}

async function runDeterministicAssetRenameFlow(input: {
  supabase: SupabaseClient;
  query: string;
  options: SpatialSearchAgentOptions;
  callbacks?: AgentRuntimeCallbacks;
}): Promise<SpatialSearchResponse | null> {
  const intent = parseDeterministicAssetRenameIntent(
    input.query,
    input.options,
  );
  if (!intent) {
    return null;
  }

  const { supabase, options, callbacks } = input;
  const trace: ToolTraceEntry[] = [];
  const state = createEmptyAssetToolState();
  const listTool = buildListModelAssetsTool(supabase, {
    selectedModelIds: options.selectedModelIds,
  });
  const renameTool = buildRenameModelAssetTool(supabase, {
    selectedModelIds: options.selectedModelIds,
    allowWrite: options.executionMode === "execute",
  });
  const batchPatchTool = buildBatchPatchModelMetadataTool(supabase, {
    selectedModelIds: options.selectedModelIds,
    allowWrite: options.executionMode === "execute",
  });

  await emitProgress(callbacks, {
    event: "status",
    data: {
      phase: "asset_rename_fast_path",
      summary: "检测到确定性的模型改名请求，优先走改名专用兜底路径",
    },
  });

  let targetModelIds: string[] = [];
  let targetSceneId: string | null = null;
  let currentDisplayName: string | null = null;

  if (intent.target.kind === "latest") {
    const listArgs = {
      modelIds: [],
      sceneIds: [],
      tags: [],
      query: "",
      startTime: null,
      endTime: null,
      limit: intent.target.count,
    };
    await emitProgress(callbacks, {
      event: "tool_call",
      data: {
        name: listTool.name,
        args: listArgs,
        summary: "先定位最新模型资产",
        round: 1,
      },
    });
    const listResult = await listTool.invoke(listArgs);
    const listText = typeof listResult === "string"
      ? listResult
      : JSON.stringify(listResult);
    const listCount = collectAssetToolResult(listTool.name, listText, state);
    const listSummary = summarizeToolResult(listTool.name, listCount);
    trace.push({
      toolName: listTool.name,
      args: listArgs,
      resultSummary: listSummary,
    });
    await emitProgress(callbacks, {
      event: "tool_result",
      data: {
        name: listTool.name,
        summary: listSummary,
        count: listCount,
        round: 1,
      },
    });

    const listedRows = state.list ?? [];
    targetModelIds = listedRows.map((item) => item.id);
    const latest = listedRows[0] ?? null;
    targetSceneId = latest?.scene_id ?? null;
    currentDisplayName = latest?.display_name?.trim() || latest?.scene_id ||
      null;
  } else {
    targetModelIds = intent.target.kind === "current"
      ? [intent.target.modelId]
      : intent.target.modelIds;
  }

  if (targetModelIds.length === 0) {
    return finalizeResponse({
      success: true,
      mode: "asset_metadata",
      intent: null,
      selection: {
        scene_id: null,
        model_id: null,
        pose_image_id: null,
        confidence: 0,
        reason: "当前请求属于模型资产元数据操作，但没有找到可改名的目标模型",
      },
      answer:
        "我没定位到可改名的目标模型。请先选中模型，或直接指定要改名的模型。",
      actions: [],
      viewer_payload: emptyViewerPayload(),
      evidence: null,
      candidates: [],
      top_candidates: [],
      selected_candidate_reason: "确定性改名兜底未找到目标模型",
      tool_trace: trace,
      asset_context: serializeAssetContext(state),
      compare_context: null,
      collection_context: null,
      creative_context: null,
      memory_graph_context: null,
    });
  }

  if (!intent.newName) {
    return finalizeResponse({
      success: true,
      mode: "asset_metadata",
      intent: null,
      selection: {
        scene_id: targetSceneId,
        model_id: targetModelIds[0] ?? null,
        pose_image_id: null,
        confidence: 0.88,
        reason: "已锁定改名目标，但用户尚未提供新名字",
      },
      answer: currentDisplayName
        ? `已定位到要改名的模型“${currentDisplayName}”，但你还没告诉我新名字。请直接说“把它改名为xxx”或“把最新模型改名为xxx”。`
        : "已定位到要改名的模型，但你还没告诉我新名字。请直接说“把它改名为xxx”。",
      actions: [],
      viewer_payload: emptyViewerPayload(),
      evidence: null,
      candidates: [],
      top_candidates: [],
      selected_candidate_reason:
        "已通过确定性规则锁定改名目标，等待用户补充新名字",
      tool_trace: trace,
      asset_context: serializeAssetContext(state),
      compare_context: null,
      collection_context: null,
      creative_context: null,
      memory_graph_context: null,
    });
  }

  const isBatchRename = targetModelIds.length > 1;
  const tool = isBatchRename ? batchPatchTool : renameTool;
  const toolArgs = isBatchRename
    ? {
      modelIds: targetModelIds,
      patch: {
        displayNameTemplate: intent.newName,
        tagsAdd: [],
        tagsRemove: [],
      },
      dryRun: options.executionMode !== "execute",
    }
    : {
      modelId: targetModelIds[0]!,
      newName: intent.newName,
      dryRun: options.executionMode !== "execute",
    };
  await emitProgress(callbacks, {
    event: "tool_call",
    data: {
      name: tool.name,
      args: toolArgs,
      summary: isBatchRename
        ? "已锁定多模型目标，开始生成批量改名结果"
        : "已锁定目标模型，开始生成改名结果",
      round: trace.length + 1,
    },
  });
  const toolResult = await tool.invoke(toolArgs);
  const resultText = typeof toolResult === "string"
    ? toolResult
    : JSON.stringify(toolResult);
  const resultCount = collectAssetToolResult(tool.name, resultText, state);
  const resultSummary = summarizeToolResult(tool.name, resultCount);
  trace.push({
    toolName: tool.name,
    args: toolArgs,
    resultSummary,
  });
  await emitProgress(callbacks, {
    event: "tool_result",
    data: {
      name: tool.name,
      summary: resultSummary,
      count: resultCount,
      round: trace.length,
    },
  });

  const operationPreview = state.operation?.preview[0] ?? null;
  const beforeName = operationPreview?.old_display_name || currentDisplayName ||
    targetSceneId;
  const afterName = operationPreview?.new_display_name || intent.newName;
  const actionText = state.operation?.dry_run
    ? (isBatchRename ? "已生成批量改名预览" : "已生成改名预览")
    : (isBatchRename ? "已完成批量改名" : "已完成改名");
  const answer = isBatchRename
    ? `${actionText}：共 ${
      state.operation?.affected_count ?? targetModelIds.length
    } 个模型将改为“${intent.newName}”。${
      state.operation?.dry_run
        ? "当前还是预览模式，如需正式写入，请用 execute 再确认一次。"
        : ""
    }`
    : `${actionText}：${beforeName ?? "该模型"} -> ${afterName}。${
      state.operation?.dry_run
        ? "当前还是预览模式，如需正式写入，请用 execute 再确认一次。"
        : ""
    }`;

  return finalizeResponse({
    success: true,
    mode: "asset_metadata",
    intent: null,
    selection: {
      scene_id: targetSceneId ?? operationPreview?.scene_id ?? null,
      model_id: targetModelIds[0] ?? null,
      pose_image_id: null,
      confidence: 0.96,
      reason: isBatchRename
        ? "已通过确定性规则锁定多模型目标并执行批量改名工具"
        : "已通过确定性规则锁定目标模型并执行改名工具",
    },
    answer,
    actions: [],
    viewer_payload: emptyViewerPayload(),
    evidence: null,
    candidates: [],
    top_candidates: [],
    selected_candidate_reason: isBatchRename
      ? "已走确定性批量改名兜底，避免只依赖 LLM 自主编排范围和写入参数"
      : "已走确定性改名兜底，避免只停留在 list_model_assets 候选列表",
    tool_trace: trace,
    asset_context: serializeAssetContext(state),
    compare_context: null,
    collection_context: null,
    creative_context: null,
    memory_graph_context: null,
  });
}
async function parseSpatialIntent(
  model: ChatOpenAI,
  query: string,
  options: SpatialSearchAgentOptions = {},
): Promise<SpatialIntent> {
  const trimmed = query.trim();
  const rewrittenQuery = trimmed
    .replace(/^(帮我|给我|请你|麻烦你|找一下|帮我找|请帮我找|最像)/, "")
    .replace(
      /(在哪|在哪里|还在吗|有没有|给我看看|帮我找出来|的空间|的场景)$/g,
      "",
    )
    .replace(/^(上周拍的|去年那次扫描里|最近拍的|最新拍的)/, "")
    .trim() || trimmed;
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
    rewrittenQuery,
    targetType: heuristicTargetType,
    objectHint: heuristicTargetType === "object" ? rewrittenQuery : null,
    locationHint:
      trimmed.match(/角落|窗边|书桌|桌面|厨房|客厅|卧室|门口|沙发旁/)?.[0] ??
        null,
    sceneHint: null,
    timeHint: heuristicTimeHint || null,
    startTime: null,
    endTime: null,
    reasoning: "优先使用规则解析，减少同步路径上的 LLM 延迟。",
  };
  const heuristicRange = normalizeExplicitTimeRange(heuristic);
  return {
    ...heuristic,
    startTime: heuristicRange.startTime,
    endTime: heuristicRange.endTime,
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
        // 避免把中文关键词直接下推到 ilike 过滤，减少 PostgREST 在大表上的慢查询风险。
        .limit(Math.max(limit * 20, 120));

      if (sceneId) {
        builder = builder.eq("scene_id", sceneId);
      }
      if (startTime) {
        builder = builder.gte("created_at", startTime);
      }
      if (endTime) {
        builder = builder.lte("created_at", endTime);
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
  const parsed = JSON.parse(payload);

  if (toolName === "pose_semantic_search") {
    const rows = poseSearchResultSchema.parse(parsed) as z.infer<
      typeof poseSearchResultSchema
    >;

    for (const row of rows) {
      const sortedFrames = row.matched_frames
        .map((frame) => ({
          image_name: frame.image_name,
          transform_matrix: frame.transform_matrix,
          similarity: Number(frame.similarity ?? 0),
          tag: frame.tag ?? null,
        }))
        .sort((a, b) => b.similarity - a.similarity);
      mergeSceneCandidate(candidates, {
        modelId: row.id,
        sceneId: row.scene_id,
        userId: row.user_id,
        description: row.description ?? "",
        objects: [],
        tags: sortedFrames.map((frame) => frame.tag ?? "").filter((value) =>
          value.length > 0
        ),
        plyPath: row.ply_path,
        previewImgPath: null,
        createdAt: row.created_at,
        metaInfo: {},
        sourceScores: {
          pose_semantic_search: Number(row.similarity),
        },
        bestPose: sortedFrames[0] ?? null,
      });
    }

    return rows.length;
  }

  const rows = sceneSearchResultSchema.parse(parsed) as z.infer<
    typeof sceneSearchResultSchema
  >;
  for (const row of rows) {
    mergeSceneCandidate(candidates, {
      modelId: row.id,
      sceneId: row.scene_id,
      userId: row.user_id,
      description: row.description ?? "",
      objects: safeArray(row.objects),
      tags: safeArray(row.tags),
      plyPath: row.ply_path,
      previewImgPath: row.preview_img_path,
      createdAt: row.created_at,
      metaInfo: row.meta_info ?? {},
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
      return ["scene_metadata_search", "pose_semantic_search"];
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

  if (
    (intent.targetType === "object" || intent.targetType === "location") &&
    usedTools.has("scene_metadata_search") &&
    evidence.candidateCount > 0
  ) {
    return false;
  }

  if (
    intent.targetType === "time" &&
    usedTools.has("recent_scene_search") &&
    evidence.candidateCount > 0
  ) {
    return false;
  }

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
  options: SpatialSearchAgentOptions = {},
  callbacks?: AgentRuntimeCallbacks,
): Promise<SelectionResult> {
  await emitProgress(callbacks, {
    event: "status",
    data: {
      phase: "selection",
      summary: `正在综合 ${
        Math.min(rankedCandidates.length, 5)
      } 个候选并生成最终回答`,
      detail: rankedCandidates.length === 0
        ? "当前没有可信候选，准备返回兜底说明"
        : `当前最高候选分数 ${(rankedCandidates[0]!.score * 100).toFixed(1)}%`,
    },
  });
  const structuredModel = model.withStructuredOutput(selectionSchema);
  const best = rankedCandidates[0] ?? null;

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
      defaultSelection: best
        ? {
          selectedSceneId: best.sceneId,
          selectedModelId: best.modelId,
          selectedPoseImageId: best.bestPose?.image_name ?? null,
          selectionReason: "综合工具检索得分最高",
          confidence: best.score,
        }
        : null,
    })),
  ]);
}

function buildDeterministicSpatialSelection(input: {
  rankedCandidates: Array<SceneCandidate & { score: number }>;
}): SelectionResult {
  const best = input.rankedCandidates[0] ?? null;
  if (!best) {
    return {
      selectedSceneId: null,
      selectedModelId: null,
      selectedPoseImageId: null,
      selectionReason: "没有检索到可信候选",
      confidence: 0,
      answer: "当前没有找到可信的空间检索结果。",
      actions: [],
    };
  }

  const poseLabel = best.bestPose?.tag ?? best.bestPose?.image_name ??
    "最佳视角";
  const objectSummary = best.objects.slice(0, 3).join("、");
  const description = best.description.trim();
  const answerSegments = [
    `我先定位到场景 ${best.sceneId}。`,
    description.length > 0 ? description : null,
    objectSummary.length > 0
      ? `该场景里识别到的相关内容包括 ${objectSummary}。`
      : null,
    best.bestPose
      ? `可以直接飞到 ${poseLabel}。`
      : "当前先打开场景，再继续手动查看。",
  ].filter((segment): segment is string => Boolean(segment && segment.trim()));

  return {
    selectedSceneId: best.sceneId,
    selectedModelId: best.modelId,
    selectedPoseImageId: best.bestPose?.image_name ?? null,
    selectionReason:
      "已按确定性检索得分选择当前最可信候选，避免因上游模型抖动阻塞简单空间查询",
    confidence: best.score,
    answer: answerSegments.join(" "),
    actions: [],
  };
}

async function executeDeterministicSpatialToolLoop(input: {
  intent: SpatialIntent;
  tools: DynamicStructuredTool[];
  callbacks?: AgentRuntimeCallbacks;
}): Promise<
  { candidates: Map<string, SceneCandidate>; trace: ToolTraceEntry[] }
> {
  const { intent, tools, callbacks } = input;
  const toolsByName = new Map(tools.map((tool) => [tool.name, tool]));
  const candidates = new Map<string, SceneCandidate>();
  const trace: ToolTraceEntry[] = [];
  const preferredTools = getPreferredToolOrder(intent);

  for (let index = 0; index < preferredTools.length; index += 1) {
    const toolName = preferredTools[index]!;
    const tool = toolsByName.get(toolName);
    if (!tool) continue;

    const args = buildToolArgs(toolName, intent);
    await emitProgress(callbacks, {
      event: "tool_call",
      data: {
        name: toolName,
        args,
        summary: `进入确定性兜底路径，执行 ${toolName}`,
        round: index + 1,
      },
    });

    const toolResult = await tool.invoke(args);
    const resultText = typeof toolResult === "string"
      ? toolResult
      : JSON.stringify(toolResult);
    const count = collectSceneCandidates(toolName, resultText, candidates);
    const resultSummary = summarizeToolResult(toolName, count);
    trace.push({
      toolName,
      args,
      resultSummary,
    });
    await emitProgress(callbacks, {
      event: "tool_result",
      data: {
        name: toolName,
        summary: `${resultSummary}，当前累计 ${candidates.size} 个候选场景`,
        count,
        round: index + 1,
      },
    });

    const evidence = summarizeCandidateEvidence(candidates, intent);
    if (
      evidence.candidateCount >= MIN_AGENT_CANDIDATES &&
      evidence.topScore >= MIN_AGENT_TOP_SCORE
    ) {
      break;
    }
  }

  return { candidates, trace };
}

async function executeAgentToolLoop(input: {
  model: ChatOpenAI;
  intent: SpatialIntent;
  tools: DynamicStructuredTool[];
  options?: SpatialSearchAgentOptions;
  callbacks?: AgentRuntimeCallbacks;
}): Promise<
  { candidates: Map<string, SceneCandidate>; trace: ToolTraceEntry[] }
> {
  const { model, intent, tools, options = {}, callbacks } = input;
  const toolsByName = new Map(tools.map((tool) => [tool.name, tool]));
  const candidates = new Map<string, SceneCandidate>();
  const trace: ToolTraceEntry[] = [];
  const agentModel = model.bindTools(tools);
  const { buildAgentContextBlock } = await import("./prompts/context.ts");
  const { getSpatialToolLoopPrompt } = await import(
    "./prompts/spatial_tool_loop.ts"
  );
  const contextBlock = buildAgentContextBlock(options);

  const messages = [
    new SystemMessage(getSpatialToolLoopPrompt(contextBlock)),
    new HumanMessage(JSON.stringify(intent)),
  ];

  for (let round = 0; round < MAX_AGENT_TOOL_ROUNDS; round += 1) {
    await emitProgress(callbacks, {
      event: "status",
      data: {
        phase: "spatial_tool_round",
        summary: `正在进行第 ${round + 1} 轮空间检索决策`,
        detail: `当前已累计 ${candidates.size} 个候选场景`,
      },
    });
    const response = await agentModel.invoke(messages);
    messages.push(response);

    let toolCalls = Array.isArray(response.tool_calls)
      ? response.tool_calls
      : [];
    if (toolCalls.length === 0) {
      const shouldForceContinue = round < MAX_AGENT_TOOL_ROUNDS - 1 &&
        shouldForceAnotherToolRound({
          intent,
          candidates,
          trace,
        });
      const forcedToolCall = shouldForceContinue
        ? buildForcedToolCall({ intent, trace })
        : null;

      if (!forcedToolCall) {
        break;
      }

      await emitProgress(callbacks, {
        event: "status",
        data: {
          phase: "spatial_tool_force_continue",
          summary: "当前证据不足，继续补充检索来源",
          detail: `自动追加 ${forcedToolCall.name} 以补齐候选和交叉证据`,
        },
      });

      messages.push(
        new SystemMessage(
          `当前证据不足，需要继续补充检索。
- 候选数不足 ${MIN_AGENT_CANDIDATES} 个、或最高分低于 ${MIN_AGENT_TOP_SCORE}、或证据来源过单时，不得直接结束。
- 下一步请补充执行 ${forcedToolCall.name}。`,
        ),
      );
      toolCalls = [forcedToolCall];
    }

    for (const toolCall of toolCalls) {
      await emitProgress(callbacks, {
        event: "tool_call",
        data: {
          name: toolCall.name,
          args: toolCall.args ?? {},
          summary: `开始执行 ${toolCall.name}`,
          round: round + 1,
        },
      });
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
      const count = collectSceneCandidates(tool.name, resultText, candidates);
      const resultSummary = summarizeToolResult(tool.name, count);
      trace.push({
        toolName: tool.name,
        args: toolCall.args ?? {},
        resultSummary,
      });
      await emitProgress(callbacks, {
        event: "tool_result",
        data: {
          name: tool.name,
          summary: `${resultSummary}，当前累计 ${candidates.size} 个候选场景`,
          count,
          round: round + 1,
        },
      });
      messages.push(
        new ToolMessage({
          tool_call_id: toolCall.id ?? toolCall.name,
          content: resultText,
        }),
      );
    }
  }

  return { candidates, trace };
}

async function executeAssetToolLoop(input: {
  model: ChatOpenAI;
  query: string;
  tools: DynamicStructuredTool[];
  options?: SpatialSearchAgentOptions;
  callbacks?: AgentRuntimeCallbacks;
}): Promise<{ trace: ToolTraceEntry[]; state: AssetToolState }> {
  const { model, query, tools, options = {}, callbacks } = input;
  const toolsByName = new Map(tools.map((tool) => [tool.name, tool]));
  const trace: ToolTraceEntry[] = [];
  const state = createEmptyAssetToolState();
  const agentModel = model.bindTools(tools);
  const today = new Date().toISOString().slice(0, 10);

  const { buildAgentContextBlock } = await import("./prompts/context.ts");
  const { getAssetToolLoopPrompt } = await import(
    "./prompts/asset_tool_loop.ts"
  );
  const contextBlock = buildAgentContextBlock(options);

  const messages = [
    new SystemMessage(getAssetToolLoopPrompt(today, contextBlock)),
    new HumanMessage(query),
  ];

  for (let round = 0; round < MAX_AGENT_TOOL_ROUNDS; round += 1) {
    await emitProgress(callbacks, {
      event: "status",
      data: {
        phase: "asset_tool_round",
        summary: `正在进行第 ${round + 1} 轮资产工具分析`,
      },
    });
    const response = await agentModel.invoke(messages);
    messages.push(response);

    const toolCalls = Array.isArray(response.tool_calls)
      ? response.tool_calls
      : [];
    if (toolCalls.length === 0) {
      break;
    }

    for (const toolCall of toolCalls) {
      await emitProgress(callbacks, {
        event: "tool_call",
        data: {
          name: toolCall.name,
          args: toolCall.args ?? {},
          summary: `开始执行 ${toolCall.name}`,
          round: round + 1,
        },
      });
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
      const resultSummary = summarizeToolResult(tool.name, count);
      trace.push({
        toolName: tool.name,
        args: toolCall.args ?? {},
        resultSummary,
      });
      await emitProgress(callbacks, {
        event: "tool_result",
        data: {
          name: tool.name,
          summary: resultSummary,
          count,
          round: round + 1,
        },
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

function finalizeResponse(
  response: SpatialSearchResponse,
): SpatialSearchResponse {
  const normalized = {
    ...response,
    session_state: response.session_state ??
      buildSessionStateFromResponse(response),
    conversation_summary: response.conversation_summary ??
      buildConversationSummaryFromResponse(response),
    follow_up: response.follow_up ?? buildFollowUpFromResponse(response),
  };
  return spatialSearchResponseSchemaUnion.parse(normalized);
}

function buildSessionStateFromResponse(
  response: SpatialSearchResponse,
): SessionState {
  const lastSelectedModelIds = new Set<string>();
  const selectedModelId = response.selection.model_id;
  if (selectedModelId) {
    lastSelectedModelIds.add(selectedModelId);
  }

  const operationPreview = response.asset_context.operation;
  const latestToolTrace = response.tool_trace[response.tool_trace.length - 1];
  if (operationPreview) {
    for (const item of operationPreview.preview) {
      if (item.model_id) {
        lastSelectedModelIds.add(item.model_id);
      }
    }
  }

  const candidateRefs = response.top_candidates.slice(0, 5).map((
    candidate,
    index,
  ) => ({
    index: index + 1,
    sceneId: candidate.scene_id,
    modelId: candidate.model_id,
    description: candidate.description,
  }));

  return {
    lastMode: response.mode,
    lastSelectedModelIds: lastSelectedModelIds.size > 0
      ? [...lastSelectedModelIds]
      : undefined,
    lastCandidateRefs: candidateRefs.length > 0 ? candidateRefs : undefined,
    lastOperationPreview: operationPreview
      ? {
        toolName: operationPreview.tool_name,
        affectedCount: operationPreview.affected_count,
        modelIds: operationPreview.preview
          .map((item) => item.model_id)
          .filter((id): id is string => Boolean(id)),
        args: latestToolTrace?.toolName === operationPreview.tool_name
          ? latestToolTrace.args
          : undefined,
      }
      : undefined,
  };
}

function buildConversationSummaryFromResponse(
  response: SpatialSearchResponse,
): string | null {
  const segments = [
    `模式: ${response.mode}`,
    response.selected_candidate_reason
      ? `判定: ${response.selected_candidate_reason}`
      : null,
    response.answer ? `结果: ${response.answer}` : null,
  ].filter((item): item is string => Boolean(item));
  return segments.length > 0 ? segments.join(" | ") : null;
}

function buildFollowUpFromResponse(
  response: SpatialSearchResponse,
): z.infer<typeof agentFollowUpSchema> {
  if (
    response.mode === "asset_metadata" &&
    response.selection.model_id &&
    response.asset_context.operation == null &&
    /还没告诉我新名字|还没告诉我新名称/.test(response.answer)
  ) {
    return {
      status: "waiting_user_input",
      kind: "rename_model",
      message: response.answer,
      input_placeholder: "例如：把它改名为宿舍书桌-03",
      suggested_replies: [
        "把它改名为宿舍书桌-03",
        "把它改名为客厅扫描-最新",
      ],
    };
  }

  if (
    response.mode === "asset_metadata" &&
    response.asset_context.operation?.requires_confirmation === true
  ) {
    return {
      status: "waiting_user_input",
      kind: "confirm_write",
      message: "当前返回的是写操作预览。确认后可继续正式执行。",
      input_placeholder: "例如：确认执行改名",
      suggested_replies: [
        "确认执行",
        "先别执行，换个名字",
      ],
    };
  }

  if (response.top_candidates.length > 1 && response.actions.length === 0) {
    return {
      status: "waiting_user_input",
      kind: "choose_candidate",
      message: "如果你想继续缩小范围，可以直接说“打开第一个”或“看第二个”。",
      input_placeholder: "例如：打开第一个",
      suggested_replies: [
        "打开第一个",
        "看第二个",
      ],
    };
  }

  return null;
}

function normalizeCompareEvidence(
  evidence: Awaited<
    ReturnType<typeof runTimeCompareAgent>
  >["comparison"]["baseline"],
): z.infer<typeof compareSceneEvidenceSchema> {
  if (!evidence) {
    return null;
  }

  return compareSceneEvidenceSchema.parse({
    ...evidence,
    matchedFrames: evidence.matchedFrames.map((frame) => ({
      imageName: frame.imageName,
      similarity: frame.similarity,
      transformMatrix: normalizeMatrix(frame.transformMatrix),
      tag: frame.tag,
    })),
  });
}

async function buildTimeCompareModeResponse(
  query: string,
  options: SpatialSearchAgentOptions,
  callbacks?: AgentRuntimeCallbacks,
): Promise<SpatialSearchResponse> {
  await emitProgress(callbacks, {
    event: "status",
    data: {
      phase: "time_compare",
      summary: "已进入时间对比模式，正在对比时间窗口中的场景差异",
    },
  });
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
  const baselineEvidence = normalizeCompareEvidence(result.comparison.baseline);
  const targetEvidence = normalizeCompareEvidence(result.comparison.target);
  const actions: VisualizationAction[] = [];
  for (const action of result.actions) {
    if (action.type === "open_scene") {
      const openSceneAction: VisualizationAction = {
        type: "open_scene",
        title: `打开${action.slot === "baseline" ? "旧版" : "新版"}场景`,
        payload: {
          sceneId: action.sceneId,
          modelId: action.modelId ?? "",
          ply: action.ply ?? null,
          poses: action.poses ?? null,
        },
      };
      actions.push(openSceneAction);
      continue;
    }

    const flyAction: VisualizationAction = {
      type: "fly_to_pose",
      title: `飞到${action.slot === "baseline" ? "旧版" : "新版"}视角`,
      payload: {
        sceneId: action.sceneId,
        imageId: action.imageName ?? null,
        matrix: normalizeMatrix(action.matrix),
      },
    };
    actions.push(flyAction);
  }

  return finalizeResponse({
    success: true,
    mode: "time_compare",
    intent: null,
    selection: {
      scene_id: result.comparison.target?.sceneId ??
        result.comparison.baseline?.sceneId ?? null,
      model_id: result.comparison.target?.modelId ??
        result.comparison.baseline?.modelId ?? null,
      pose_image_id: result.comparison.target?.matchedFrames[0]?.imageName ??
        result.comparison.baseline?.matchedFrames[0]?.imageName ??
        null,
      confidence: result.comparison.target?.similarity ??
        result.comparison.baseline?.similarity ?? 0,
      reason: selectedReason,
    },
    answer: result.answer,
    actions,
    viewer_payload: emptyViewerPayload(),
    evidence: {
      baseline: baselineEvidence,
      target: targetEvidence,
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
      baseline: baselineEvidence,
      target: targetEvidence,
      diff: result.comparison.diff,
      windows: result.intent,
    },
    collection_context: null,
    creative_context: null,
    memory_graph_context: null,
  });
}

async function buildCreativeModeResponse(input: {
  supabase: SupabaseClient;
  query: string;
  options: SpatialSearchAgentOptions;
  callbacks?: AgentRuntimeCallbacks;
}): Promise<SpatialSearchResponse> {
  await emitProgress(input.callbacks, {
    event: "status",
    data: {
      phase: "creative_prepare",
      summary: "已进入创作模式，正在整理故事素材和时间线",
    },
  });
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
  await emitProgress(input.callbacks, {
    event: "status",
    data: {
      phase: "creative_outline",
      summary: `已生成 ${outline.outline.length} 段创作大纲`,
      detail: `本次整理了 ${storyContext.model_count} 个模型素材`,
    },
  });
  const task = input.options.executionMode === "execute"
    ? await enqueueCreativeTask(input.supabase, {
      query: input.query,
      modelIds,
      outline,
      currentSceneId: input.options.currentSceneId,
    })
    : null;

  return finalizeResponse({
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
    selected_candidate_reason:
      "创作模式不返回空间候选，而是返回素材上下文和大纲",
    tool_trace: [
      {
        toolName: "prepare_story_context",
        args: { modelIds },
        resultSummary:
          `prepare_story_context 读取 ${storyContext.model_count} 个模型`,
      },
      {
        toolName: "generate_story_outline",
        args: { query: input.query },
        resultSummary:
          `generate_story_outline 生成 ${outline.outline.length} 段大纲`,
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
  });
}

async function buildMemoryGraphModeResponse(input: {
  supabase: SupabaseClient;
  query: string;
  options: SpatialSearchAgentOptions;
  callbacks?: AgentRuntimeCallbacks;
}): Promise<SpatialSearchResponse> {
  await emitProgress(input.callbacks, {
    event: "status",
    data: {
      phase: "memory_graph",
      summary: "已进入长期记忆模式，正在汇总趋势、缺失模式和变化时间线",
    },
  });
  const focusModelId = input.options.currentModelId ??
    input.options.selectedModelIds?.[0];
  if (!focusModelId) {
    throw new Error("长期记忆模式需要当前模型或至少一个已选模型");
  }

  const trend = await getRecentPlaceTrend(input.supabase, {
    modelId: focusModelId,
    lookback: 5,
  });
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

  return finalizeResponse({
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
    answer: `${trend.summary}${
      missing.missing ? ` ${missing.summary}` : ""
    } ${timeline.summary}`,
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
  });
}

export async function runSpatialSearchAgent(
  query: string,
  options: SpatialSearchAgentOptions = {},
  callbacks: AgentRuntimeCallbacks = {},
): Promise<SpatialSearchResponse> {
  const env = ensureRuntimeEnv();
  const supabase = createSupabaseAdminClient(env);
  const model = createChatModel(env);
  await emitProgress(callbacks, {
    event: "status",
    data: {
      phase: "bootstrap",
      summary: "Agent 已收到请求，正在初始化检索上下文",
      detail: `执行模式：${options.executionMode ?? "preview"}`,
    },
  });
  if (isDirectReplyQuery(query)) {
    const directReplyIntent: SpatialIntent = {
      rewrittenQuery: query.trim(),
      targetType: "scene",
      objectHint: null,
      locationHint: null,
      sceneHint: null,
      timeHint: null,
      startTime: null,
      endTime: null,
      reasoning: "当前输入属于闲聊问候，直接返回自然语言应答。",
    };
    return finalizeResponse({
      success: true,
      mode: "spatial_search",
      intent: directReplyIntent,
      selection: {
        scene_id: null,
        model_id: null,
        pose_image_id: null,
        confidence: 1,
        reason: "当前输入属于闲聊问候，无需进入检索链路",
      },
      answer: /谢/.test(query)
        ? "不客气。"
        : "你好，我在。你可以直接告诉我想找的物体、位置或场景。",
      actions: [],
      viewer_payload: emptyViewerPayload(),
      evidence: null,
      candidates: [],
      top_candidates: [],
      selected_candidate_reason: "已走闲聊直答兜底，避免误进空间检索链路",
      tool_trace: [],
      asset_context: serializeAssetContext(createEmptyAssetToolState()),
      compare_context: null,
      collection_context: null,
      creative_context: null,
      memory_graph_context: null,
    });
  }

  let mode: AgentMode;
  try {
    mode = await classifyAgentMode(model, query, options);
  } catch (error) {
    if (!shouldPreferHeuristicSpatialRoute(query)) {
      throw error;
    }
    await emitProgress(callbacks, {
      event: "status",
      data: {
        phase: "route_fallback",
        summary: "模式分类依赖的上游模型暂时不可用，已回退到确定性空间检索",
        detail: error instanceof Error ? error.message : String(error),
      },
    });
    mode = "spatial_search";
  }
  await emitProgress(callbacks, {
    event: "status",
    data: {
      phase: "route",
      summary: `已完成模式判断：${mode}`,
    },
  });

  if (mode === "time_compare") {
    return await buildTimeCompareModeResponse(query, options, callbacks);
  }

  if (mode === "creative") {
    return await buildCreativeModeResponse({
      supabase,
      query,
      options,
      callbacks,
    });
  }

  if (mode === "memory_graph") {
    return await buildMemoryGraphModeResponse({
      supabase,
      query,
      options,
      callbacks,
    });
  }

  if (mode === "asset_metadata") {
    const replayedWriteResponse = await replayPendingAssetWriteIfNeeded({
      supabase,
      query,
      options,
      callbacks,
    });
    if (replayedWriteResponse) {
      return replayedWriteResponse;
    }

    const deterministicAssetResponse = await runDeterministicAssetRenameFlow({
      supabase,
      query,
      options,
      callbacks,
    });
    if (deterministicAssetResponse) {
      return deterministicAssetResponse;
    }

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
      callbacks,
    });

    return finalizeResponse({
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
    });
  }

  const embeddings = createEmbeddingsModel(env);

  await emitProgress(callbacks, {
    event: "status",
    data: {
      phase: "intent",
      summary: "正在解析空间意图和时间约束",
    },
  });
  const intent = await parseSpatialIntent(model, query, options);
  await emitProgress(callbacks, {
    event: "status",
    data: {
      phase: "intent_done",
      summary: `意图解析完成，目标类型为 ${intent.targetType}`,
      detail: intent.rewrittenQuery,
    },
  });
  const tools = [
    await buildPoseTool(supabase, embeddings),
    await buildSceneTool(supabase),
    await buildRecentSceneTool(supabase),
  ];
  let candidateMap: Map<string, SceneCandidate>;
  let trace: ToolTraceEntry[];
  let usedDeterministicFallback = false;

  try {
    if (shouldPreferHeuristicSpatialRoute(query)) {
      await emitProgress(callbacks, {
        event: "status",
        data: {
          phase: "spatial_fast_path",
          summary: "当前请求较简单，优先走确定性空间检索路径",
          detail: "减少对上游模型路由和工具编排的依赖，避免简单查询被 503 阻塞",
        },
      });
      ({ candidates: candidateMap, trace } =
        await executeDeterministicSpatialToolLoop({
          intent,
          tools,
          callbacks,
        }));
      usedDeterministicFallback = true;
    } else {
      ({ candidates: candidateMap, trace } = await executeAgentToolLoop({
        model,
        intent,
        tools,
        options,
        callbacks,
      }));
    }
  } catch (error) {
    await emitProgress(callbacks, {
      event: "status",
      data: {
        phase: "spatial_fallback",
        summary: "空间工具编排阶段出现异常，已回退到确定性检索路径",
        detail: error instanceof Error ? error.message : String(error),
      },
    });
    ({ candidates: candidateMap, trace } =
      await executeDeterministicSpatialToolLoop({
        intent,
        tools,
        callbacks,
      }));
    usedDeterministicFallback = true;
  }

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

  let selection: SelectionResult;
  if (rankedCandidates.length === 0) {
    selection = buildDeterministicSpatialSelection({ rankedCandidates });
  } else if (usedDeterministicFallback) {
    selection = buildDeterministicSpatialSelection({ rankedCandidates });
  } else {
    try {
      selection = await selectBestResult(
        model,
        intent,
        rankedCandidates,
        suggestedActions,
        options,
        callbacks,
      );
    } catch (error) {
      await emitProgress(callbacks, {
        event: "status",
        data: {
          phase: "selection_fallback",
          summary: "最终裁决阶段出现异常，已改用确定性候选排序结果",
          detail: error instanceof Error ? error.message : String(error),
        },
      });
      selection = buildDeterministicSpatialSelection({ rankedCandidates });
      usedDeterministicFallback = true;
    }
  }

  await emitProgress(callbacks, {
    event: "status",
    data: {
      phase: "finalize",
      summary: rankedCandidates.length === 0
        ? "未找到可信候选，准备返回兜底说明"
        : `已选定场景 ${
          selection.selectedSceneId ?? rankedCandidates[0]!.sceneId
        }`,
      detail: selection.selectionReason,
    },
  });

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

  return finalizeResponse({
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
    tool_trace: usedDeterministicFallback
      ? [
        ...trace,
        {
          toolName: "deterministic_spatial_fallback",
          args: { query },
          resultSummary: "已使用确定性空间检索兜底，避免简单查询依赖上游模型",
        },
      ]
      : trace,
    asset_context: serializeAssetContext(createEmptyAssetToolState()),
    compare_context: null,
    collection_context: null,
    creative_context: null,
    memory_graph_context: null,
  });
}
