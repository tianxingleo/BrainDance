import fs from 'fs';

let code = fs.readFileSync('supabase/functions/_shared/agent-core/spatialAgent.ts', 'utf8');

const regex1 = /export function parseDeterministicAssetRenameIntent[\s\S]*?return null;\r?\n\}/m;
const regex2 = /async function runDeterministicAssetRenameFlow[\s\S]*?return finalizeResponse\(\{[\s\S]*?\}\);\r?\n\}/m;

let changed = false;

if (regex1.test(code)) {
    code = code.replace(regex1, '');
    changed = true;
    console.log("Removed parseDeterministicAssetRenameIntent");
}

if (regex2.test(code)) {
    code = code.replace(regex2, '');
    changed = true;
    console.log("Removed runDeterministicAssetRenameFlow");
}

const callerRegex = /\s*const deterministicAssetResponse = await runDeterministicAssetRenameFlow\(\{[\s\S]*?if \(deterministicAssetResponse\) \{\r?\n\s*return deterministicAssetResponse;\r?\n\s*\}/m;
if (callerRegex.test(code)) {
    code = code.replace(callerRegex, '');
    changed = true;
    console.log("Removed caller logic");
}

if (changed) {
    fs.writeFileSync('supabase/functions/_shared/agent-core/spatialAgent.ts', code);
}
