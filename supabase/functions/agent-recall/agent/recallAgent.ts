import { runSpatialSearchAgent } from "../../_shared/agent-core/spatialAgent.ts";
import {
  type AgentRecallResponse,
  type AgentRecallStreamEvent,
  agentRecallStreamEventSchema,
  agentRecallResponseSchema,
  type agentRecallActionSchema,
} from "../schemas/response.ts";

type RecallAgentOptions = {
  onEvent?: (event: AgentRecallStreamEvent) => void | Promise<void>;
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

export async function runRecallAgent(
  query: string,
  options: RecallAgentOptions = {},
): Promise<AgentRecallResponse> {
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

  const spatialResult = await runSpatialSearchAgent(query, {
    executionMode: "preview" // default to preview for safety
  });

  await emitEvent(options, {
    event: "tool_result",
    data: {
      name: "shared_agent_core",
      status: spatialResult.success ? "success" : "error",
      result: {
        mode: spatialResult.mode,
        candidates_count: spatialResult.candidates?.length ?? 0
      }
    },
  });

  await emitEvent(options, {
    event: "thinking",
    data: {
      content: spatialResult.candidates?.length === 0 && spatialResult.mode === 'spatial_search'
        ? "当前没有命中可信场景，需要明确告诉用户暂无结果。"
        : "已经拿到候选结果，接下来整理证据和所需的界面动作。",
    },
  });

  // Map actions
  const mappedActions: any[] = [];
  for (const a of spatialResult.actions) {
    if (a.type === 'open_model') {
      mappedActions.push({
        type: 'open_scene',
        ...a.payload
      });
    } else if (a.type === 'fly_to_pose') {
      mappedActions.push({
        type: 'fly_to_pose',
        ...a.payload
      });
    }
  }

  const response: AgentRecallResponse = {
    answer: spatialResult.answer || "已处理您的请求。",
    evidence: spatialResult.mode === 'spatial_search' && spatialResult.candidates && spatialResult.candidates.length > 0 ? {
      sceneId: spatialResult.candidates[0].scene_id,
      similarity: spatialResult.candidates[0].score,
      matchedFrames: spatialResult.candidates[0].pose_image_id ? [{
        imageName: spatialResult.candidates[0].pose_image_id,
        similarity: spatialResult.candidates[0].score,
        transformMatrix: spatialResult.viewer_payload.matrix
      }] : []
    } : null,
    actions: mappedActions,
    top_candidates: spatialResult.candidates?.map(c => ({
      sceneId: c.scene_id,
      similarity: c.score,
      description: c.description
    })) || [],
    selected_candidate_reason: spatialResult.selection?.reason || "",
  };

  const parsed = agentRecallResponseSchema.parse(response);
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
