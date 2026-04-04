import fs from 'fs';

let code = fs.readFileSync('supabase/functions/_shared/agent-core/spatialAgent.ts', 'utf8');

const regex = /<<<<<<< Updated upstream[\s\S]*?>>>>>>> Stashed changes/;
const replacement = `export function isDirectReplyQuery(query: string): boolean {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return false;

  return /^(你好|您好|hello|hi|thanks|thank you|谢谢|多谢|辛苦了|再见|拜拜)[！!,.，。？?]*$/.test(normalized);
}`;

if (regex.test(code)) {
    code = code.replace(regex, replacement);
    fs.writeFileSync('supabase/functions/_shared/agent-core/spatialAgent.ts', code);
    console.log("Resolved conflict successfully");
} else {
    console.log("Conflict not found");
}
