import fs from 'fs';

let code = fs.readFileSync('supabase/functions/_shared/agent-core/spatialAgent.ts', 'utf8');

function findBlock(startRegex, endRegex) {
    let matchStart = code.match(startRegex);
    if (!matchStart) return null;
    let start = matchStart.index;
    
    let sliced = code.slice(start);
    let matchEnd = sliced.match(endRegex);
    if (!matchEnd) return null;
    let end = start + matchEnd.index + matchEnd[0].length;
    return { start, end };
}

function removeFunction(funcHeaderRegex) {
    let match = code.match(funcHeaderRegex);
    if (!match) {
        console.log("Could not find regex", funcHeaderRegex);
        return;
    }
    let start = match.index;
    let braceStart = code.indexOf('{', start);
    if (braceStart === -1) return;
    
    let braceCount = 1;
    let i = braceStart + 1;
    while (braceCount > 0 && i < code.length) {
        if (code[i] === '{') braceCount++;
        else if (code[i] === '}') braceCount--;
        i++;
    }
    
    if (braceCount === 0) {
        console.log("Removed from", start, "to", i);
        code = code.slice(0, start) + code.slice(i);
    }
}

// 1. Remove heuristic functions
removeFunction(/export function shouldPreferHeuristicSpatialRoute/);
removeFunction(/export function parseDeterministicAssetRenameIntent/);
removeFunction(/async function runDeterministicAssetRenameFlow/);
removeFunction(/function buildDeterministicSpatialSelection/);
removeFunction(/async function executeDeterministicSpatialToolLoop/);

// 2. ClassifyAgentMode replacement
let classifyStartRegex = /async function classifyAgentMode\(/;
let contextRegex = /const \{ buildAgentContextBlock \} = await import\("\.\/prompts\/context\.ts"\);/;
let block = findBlock(classifyStartRegex, contextRegex);
if (block) {
    let replace = `async function classifyAgentMode(
  model: BaseChatModel,
  query: string,
  options: SpatialSearchAgentOptions = {},
): Promise<AgentMode> {
  const currentMode = options.currentMode ?? null;

  if (isDirectReplyQuery(query)) {
    return "spatial_search";
  }

  if (currentMode === "compare") {
    return "time_compare";
  }
  if (currentMode === "batch_edit" || currentMode === "collection") {
    return "asset_metadata";
  }

  const { buildAgentContextBlock } = await import("./prompts/context.ts");`;
    code = code.slice(0, block.start) + replace + code.slice(block.end);
    console.log("Replaced classifyAgentMode");
} else {
    console.log("Could not find classifyAgentMode block");
}

// 3. remove deterministic asset rename caller
let callerRegexStart = /const deterministicAssetResponse = await runDeterministicAssetRenameFlow\(\{/;
let callerRegexEnd = /return deterministicAssetResponse;\s*\}/;
let callerBlock = findBlock(callerRegexStart, callerRegexEnd);
if (callerBlock) {
    code = code.slice(0, callerBlock.start) + code.slice(callerBlock.end);
    console.log("Removed deterministicAssetResponse caller");
} else {
    console.log("Could not find deterministicAssetResponse caller");
}

// 4. remove deterministic tool loop fallback
let tryStart = /try \{\s*if \(shouldPreferHeuristicSpatialRoute\(query\)\) \{/;
let tryEndStr = /\}\)\);\s*\}/;
let tryBlock = findBlock(tryStart, tryEndStr);
if (tryBlock) {
    let replace = `try {
    ({ candidates: candidateMap, trace } = await executeAgentToolLoop({
      model,
      intent,
      tools,
      options,
      callbacks,
    }));`;
    code = code.slice(0, tryBlock.start) + replace + code.slice(tryBlock.end);
    console.log("Replaced deterministic agent fallback");
} else {
    console.log("Could not find try block for deterministic fallback");
}

// 5. Rewrite parseSpatialIntent
let parseStartRegex = /async function parseSpatialIntent\(/;
let parseEndRegex = /endTime: heuristicRange\.endTime,\s*\};\s*\}/;
let pBlock = findBlock(parseStartRegex, parseEndRegex);
if (pBlock) {
    let replace = `async function parseSpatialIntent(
  model: BaseChatModel,
  query: string,
  options: SpatialSearchAgentOptions = {},
): Promise<SpatialIntent> {
  const { buildAgentContextBlock } = await import("./prompts/context.ts");
  const { getSpatialIntentPrompt } = await import("./prompts/spatial_intent.ts");
  const contextBlock = buildAgentContextBlock(options);
  const today = new Date().toISOString().slice(0, 10);

  const structuredModel = model.withStructuredOutput(spatialIntentSchema);
  const result = await structuredModel.invoke([
    new SystemMessage(getSpatialIntentPrompt(today, contextBlock)),
    new HumanMessage(query),
  ]);

  const timeRange = normalizeExplicitTimeRange(result);
  return {
    ...result,
    startTime: timeRange.startTime,
    endTime: timeRange.endTime,
  };
}`;
    code = code.slice(0, pBlock.start) + replace + code.slice(pBlock.end);
    console.log("Replaced parseSpatialIntent");
} else {
    console.log("Could not find parseSpatialIntent block");
}

let catchReplaceCount = 0;
// 6. Fix \`shouldPreferHeuristicSpatialRoute\` calls in \`try ... catch ...\` block
while(code.match(/\} catch \(error\) \{\s*if \(\!shouldPreferHeuristicSpatialRoute\(query\)\) \{\s*throw error;\s*\}/)) {
    code = code.replace(/\} catch \(error\) \{\s*if \(\!shouldPreferHeuristicSpatialRoute\(query\)\) \{\s*throw error;\s*\}/, `} catch (error) {\n    throw error;`);
    catchReplaceCount++;
}
console.log("Removed shouldPreferHeuristicSpatialRoute from catch block:", catchReplaceCount);

// 7. Fix fallback selection block 
let catchFallbackStart = /\} catch \(error\) \{\s*await emitProgress\(callbacks, \{\s*event: "status",\s*data: \{\s*phase: "selection_fallback"[\s\S]*?usedDeterministicFallback = true;\s*\}/;
if (code.match(catchFallbackStart)) {
    code = code.replace(catchFallbackStart, `} catch (error) {
      await emitProgress(callbacks, {
        event: "status",
        data: {
          phase: "selection_fallback",
          summary: "最终裁决阶段出现异常",
          detail: error instanceof Error ? error.message : String(error),
        },
      });
      throw error;
    }`);
    console.log("Removed fallback in selection catch");
}

let ifFallback = /if \(rankedCandidates\.length === 0\) \{\s*selection = buildDeterministicSpatialSelection[\s\S]*?\} else if \(usedDeterministicFallback\) \{\s*selection = buildDeterministicSpatialSelection[\s\S]*?\} else \{/;
if (code.match(ifFallback)) {
    code = code.replace(ifFallback, `if (rankedCandidates.length === 0) {
    throw new Error("No candidates found");
  } else {`);
    console.log("Removed fallback in selection if blocks");
}

fs.writeFileSync('supabase/functions/_shared/agent-core/spatialAgent.ts', code);
console.log("Done");