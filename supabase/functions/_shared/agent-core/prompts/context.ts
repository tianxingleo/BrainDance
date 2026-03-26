import { SpatialSearchAgentOptions } from "../spatialAgent.ts";

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
  }

  if (options.conversationSummary) {
    parts.push(`\n=== 历史会话摘要 ===`);
    parts.push(`${options.conversationSummary}`);
  }

  parts.push("=========================\n");

  return parts.join("\n");
}
