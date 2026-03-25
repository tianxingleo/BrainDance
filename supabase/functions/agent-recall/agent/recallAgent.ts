import { runSpatialSearchAgent } from "../../_shared/agent-core/spatialAgent.ts";
import {
  type AgentRecallResponse,
  type AgentRecallStreamEvent,
  agentRecallStreamEventSchema,
  agentRecallResponseSchema,
} from "../schemas/response.ts";

type SpatialAgentResult = Awaited<ReturnType<typeof runSpatialSearchAgent>>;
type RecallAgentExecutor = typeof runSpatialSearchAgent;

type RecallAgentOptions = {
  onEvent?: (event: AgentRecallStreamEvent) => void | Promise<void>;
  execute?: RecallAgentExecutor;
};

async function emitEvent(
  options: RecallAgentOptions,
  event: AgentRecallStreamEvent,
) {
  if (!options.onEvent) {
    return;
  }
  await options.onEvent(agentRecallStreamEventSchema.parse(event));
}

function readString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function readNullableString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function summarizeSpatialResult(result: SpatialAgentResult) {
  return {
    mode: result.mode,
    success: result.success,
    selectedSceneId: result.selection?.scene_id ?? null,
    selectedReason: result.selection?.reason ?? "",
    candidateCount: result.candidates?.length ?? 0,
    topCandidates: (result.candidates ?? []).slice(0, 3).map((candidate) => ({
      sceneId: candidate.scene_id,
      similarity: candidate.score,
      description: candidate.description,
    })),
  };
}

function mapRecallActions(
  result: SpatialAgentResult,
): AgentRecallResponse["actions"] {
  const mappedActions: AgentRecallResponse["actions"] = [];
  for (const action of result.actions) {
    if (action.type === "open_model") {
      const payload = action.payload as Record<string, unknown>;
      mappedActions.push({
        type: "open_scene",
        sceneId: readString(payload.sceneId),
        modelId: readNullableString(payload.modelId),
        ply: readNullableString(payload.ply),
        poses: readNullableString(payload.poses),
      });
      continue;
    }
    if (action.type === "fly_to_pose") {
      const payload = action.payload as Record<string, unknown>;
      mappedActions.push({
        type: "fly_to_pose",
        sceneId: readString(payload.sceneId),
        imageName: readString(payload.imageId) || undefined,
        matrix: payload.matrix ?? null,
      });
    }
  }
  return mappedActions;
}

function buildRecallResponse(result: SpatialAgentResult): AgentRecallResponse {
  const topCandidate = result.candidates?.[0];
  return {
    answer: result.answer || "已处理您的请求。",
    evidence: result.mode === "spatial_search" && topCandidate
      ? {
        sceneId: topCandidate.scene_id,
        similarity: topCandidate.score,
        matchedFrames: topCandidate.pose_image_id
          ? [{
            imageName: topCandidate.pose_image_id,
            similarity: topCandidate.score,
            transformMatrix: result.viewer_payload.matrix,
          }]
          : [],
      }
      : null,
    actions: mapRecallActions(result),
    top_candidates: (result.candidates ?? []).map((candidate) => ({
      sceneId: candidate.scene_id,
      similarity: candidate.score,
      description: candidate.description,
    })),
    selected_candidate_reason: result.selection?.reason || "",
  };
}

export async function runRecallAgent(
  query: string,
  options: RecallAgentOptions = {},
): Promise<AgentRecallResponse> {
  const execute = options.execute ?? runSpatialSearchAgent;
  await emitEvent(options, {
    event: "plan",
    data: {
      title: "多工具检索 Agent 执行计划",
      steps: [
        "理解用户意图并路由（空间检索或资产管理）",
        "执行多工具检索或资产对比操作",
        "提候选与裁决",
        "生成最终回答并返回可执行动作",
      ],
    },
  });

  await emitEvent(options, {
    event: "thinking",
    data: {
      content: "开始调用统一 Agent 核心处理请求...",
    },
  });

  await emitEvent(options, {
    event: "tool_call",
    data: {
      name: "run_spatial_search_agent",
      args: {
        query,
        executionMode: "preview",
      },
    },
  });

  const spatialResult = await execute(query, {
    executionMode: "preview", // preview 默认只返回动作建议，不直接执行副作用
  });

  await emitEvent(options, {
    event: "tool_result",
    data: {
      name: "run_spatial_search_agent",
      status: !spatialResult.success
          ? "error"
          : (spatialResult.candidates?.length ?? 0) == 0
          ? "empty"
          : "success",
      result: summarizeSpatialResult(spatialResult),
    },
  });

  await emitEvent(options, {
    event: "thinking",
    data: {
      content: spatialResult.candidates?.length === 0 &&
          spatialResult.mode === "spatial_search"
        ? "当前没有命中可信场景，需要明确告诉用户暂无结果。"
        : "已经拿到候选结果，接下来整理证据和所需的界面动作。",
    },
  });
  const parsed = agentRecallResponseSchema.parse(
    buildRecallResponse(spatialResult),
  );
  await emitEvent(options, {
    event: "message",
    data: {
      delta: parsed.answer,
    },
  });
  await emitEvent(options, {
    event: "done",
    data: parsed,
  });

  return parsed;
}
