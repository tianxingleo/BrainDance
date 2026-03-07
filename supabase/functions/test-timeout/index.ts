import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
serve(async (req) => {
  await new Promise(r => setTimeout(r, 2500));
  return new Response(JSON.stringify({ok: true}));
});
