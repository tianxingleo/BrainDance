import { SpatialSearchAgentOptions } from "../spatialAgent.ts";
import { type LongTermMemory } from "../longTermMemory.ts";

export function buildAgentContextBlock(
  options: SpatialSearchAgentOptions,
): string {
  const parts: string[] = [];

  parts.push("=== 当前产品上下文/UI 状态 ===");

  if (options.executionMode) {
    parts.push(
      `- 当前执行模式: ${options.executionMode} (${
        options.executionMode === "preview"
          ? "处于安全预览模式，不允许直接执行写操作，应调用工具产生预览"
          : "处于执行模式，如果用户确认，可正式调用并提交写入"
      })`,
    );
  }

  if (options.selectedModelIds && options.selectedModelIds.length > 0) {
    const ids = options.selectedModelIds.map((id) => `"${id}"`).join(", ");
    parts.push(
      `- 工作台用户当前已选中的模型 IDs: [${ids}] (如果用户要求“批量操作”、“对这些/这个模型”进行操作，请默认只操作这些 ID)`,
    );
  } else {
    parts.push(`- 工作台用户当前已选中的模型 IDs: 无`);
  }

  if (options.currentSceneId) {
    parts.push(`- 当前正在查看的 Scene ID: ${options.currentSceneId}`);
  }

  if (options.currentModelId) {
    parts.push(`- 当前 Viewer 中打开的模型 ID: ${options.currentModelId}`);
  }

  if (options.currentMode) {
    parts.push(`- 当前工作台模式: ${options.currentMode}`);
  }

  const candidateRefs = options.sessionState?.lastCandidateRefs ?? [];
  if (candidateRefs.length > 0) {
    parts.push("- 当前候选引用:");
    for (const item of candidateRefs) {
      parts.push(
        `  ${item.index}. ${item.sceneId} / ${item.modelId} / ${item.description}`,
      );
    }
  } else if (
    options.candidateSceneIds && options.candidateSceneIds.length > 0
  ) {
    parts.push(
      `- 当前历史候选列表包含了 ${options.candidateSceneIds.length} 个模型 (如果用户说“上一个”、“第二个”，可能指代这个列表中的项目)`,
    );
  }

  if (options.sessionState?.lastMode) {
    parts.push(`- 上一轮模式: ${options.sessionState.lastMode}`);
  }

  if (
    options.sessionState?.lastSelectedModelIds &&
    options.sessionState.lastSelectedModelIds.length > 0
  ) {
    parts.push(
      `- 上一轮关联模型 IDs: [${
        options.sessionState.lastSelectedModelIds.map((id) => `"${id}"`).join(
          ", ",
        )
      }]`,
    );
  }

  if (options.sessionState?.lastOperationPreview) {
    parts.push(
      `- 上一轮操作预览: ${options.sessionState.lastOperationPreview.toolName}，影响 ${options.sessionState.lastOperationPreview.affectedCount} 个对象`,
    );
    if (
      options.sessionState.lastOperationPreview.modelIds &&
      options.sessionState.lastOperationPreview.modelIds.length > 0
    ) {
      parts.push(
        `- 上一轮操作目标模型 IDs: [${
          options.sessionState.lastOperationPreview.modelIds.map((id) =>
            `"${id}"`
          ).join(", ")
        }]`,
      );
    }
    if (options.sessionState.lastOperationPreview.args) {
      parts.push(
        `- 上一轮操作参数: ${
          JSON.stringify(options.sessionState.lastOperationPreview.args)
        }`,
      );
    }
  }

  if (options.shortTermMemory) {
    const mem = options.shortTermMemory;
    if (mem.entities.length > 0) {
      parts.push("\n=== 短期记忆：实体追踪 ===");
      parts.push(
        '以下是本次会话中提到过的实体。当用户使用指代词（"那个"、"上面的"、"刚才的"、"这个模型"）时，优先指代 mentionedAt 最大的实体：',
      );
      for (const e of mem.entities) {
        parts.push(
          `  - [${e.kind}] ${e.label} (id: ${e.id}, 第${e.mentionedAt}轮)`,
        );
      }
    }
    const prefs = mem.preferences;
    if (prefs.regions?.length || prefs.assetTypes?.length || prefs.timeRange) {
      parts.push("\n=== 短期记忆：用户偏好 ===");
      parts.push("本次会话中观察到的用户搜索偏好（可用于优化搜索参数）：");
      if (prefs.regions?.length) {
        parts.push(`  - 偏好区域: ${prefs.regions.join("、")}`);
      }
      if (prefs.assetTypes?.length) {
        parts.push(`  - 偏好资产类型: ${prefs.assetTypes.join("、")}`);
      }
      if (prefs.timeRange) {
        parts.push(`  - 偏好时间范围: ${prefs.timeRange}`);
      }
    }
  }

  if (options.longTermMemory) {
    const ltm = options.longTermMemory;
    const hasPrefs = ltm.preferredRegions.length > 0 ||
      ltm.preferredAssetTypes.length > 0 ||
      ltm.preferredObjects.length > 0 ||
      ltm.preferredTimeRanges.length > 0;
    if (hasPrefs || ltm.recentSearches.length > 0) {
      parts.push("\n=== 长期记忆：用户历史偏好 ===");
      parts.push(
        "以下是该用户跨会话积累的搜索偏好。当搜索结果有多个候选时，可优先展示符合用户历史偏好的结果，并在回答中说明「根据您的历史偏好，我优先搜索了...」：",
      );
      if (ltm.preferredRegions.length > 0) {
        parts.push(`  - 常搜区域: ${ltm.preferredRegions.join("、")}`);
      }
      if (ltm.preferredAssetTypes.length > 0) {
        parts.push(`  - 常搜资产类型: ${ltm.preferredAssetTypes.join("、")}`);
      }
      if (ltm.preferredObjects.length > 0) {
        parts.push(`  - 常搜物体: ${ltm.preferredObjects.join("、")}`);
      }
      if (ltm.preferredTimeRanges.length > 0) {
        parts.push(`  - 常用时间范围: ${ltm.preferredTimeRanges.join("、")}`);
      }
      if (ltm.recentSearches.length > 0) {
        parts.push(`  - 最近搜索 (共 ${ltm.searchCount} 次):`);
        for (const s of ltm.recentSearches.slice(-5)) {
          parts.push(`    · "${s.query}" → ${s.topResultSummary}`);
        }
      }
    }
  }

  if (options.conversationSummary) {
    parts.push(`\n=== 历史会话摘要 ===`);
    parts.push(`${options.conversationSummary}`);
  }

  parts.push("\n=== Agent 工作约束 ===");
  parts.push("- 先判断是否真的需要工具；如果工具不适合当前问题，应直接回答、澄清或引导。");
  parts.push("- 不要为了显得主动而重复调用相同工具或相同参数。");
  parts.push("- 当已有结果足以支撑回答、预览或确认时，应主动停止工具循环。");

  parts.push("=========================\n");

  return parts.join("\n");
}
