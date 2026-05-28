import fs from 'fs';

let code = fs.readFileSync('supabase/functions/_shared/agent-core/spatialAgent.ts', 'utf8');

const regex = /async function parseSpatialIntent\([\s\S]*?return \{\r?\n\s*\.\.\.heuristic,\r?\n\s*startTime: heuristicRange\.startTime,\r?\n\s*endTime: heuristicRange\.endTime,\r?\n\s*\};\r?\n\}/m;

const replacement = `async function parseSpatialIntent(
  model: ChatOpenAI,
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

if (regex.test(code)) {
    code = code.replace(regex, replacement);
    fs.writeFileSync('supabase/functions/_shared/agent-core/spatialAgent.ts', code);
    console.log("Replaced parseSpatialIntent successfully");
} else {
    console.log("Could not find parseSpatialIntent");
}
