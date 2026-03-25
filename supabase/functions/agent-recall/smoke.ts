import {
  assertEquals,
  assertExists,
} from "https://deno.land/std@0.168.0/testing/asserts.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "http://127.0.0.1:54321";
const SUPABASE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const AGENT_RECALL_CASES = [
  "黑色耳机在哪",
  "窗边那个台灯还在吗",
  "上周拍的红色杯子",
  "最像厨房角落堆着纸箱的空间",
];

Deno.test({
  name: "agent-recall smoke - 固定查询至少返回 answer、mode，并保持稳定动作协议",
  async fn() {
    if (!SUPABASE_KEY) {
      return;
    }

    for (const query of AGENT_RECALL_CASES) {
      const resp = await fetch(`${SUPABASE_URL}/functions/v1/agent-recall`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${SUPABASE_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      assertEquals(resp.status < 500, true, `${query} 不应返回 5xx`);
      const data = await resp.json();
      assertEquals(typeof data.answer, "string", `${query} 应返回 answer`);
      assertEquals(typeof data.mode, "string", `${query} 应返回 mode`);
      assertExists(data.actions, `${query} 应返回 actions`);
      assertEquals(
        Array.isArray(data.actions),
        true,
        `${query} 的 actions 必须是数组`,
      );

      if (data.actions.length > 0) {
        const actionTypes = data.actions.map((action: { type?: string }) =>
          action.type
        );
        assertEquals(
          actionTypes.some((type: string) =>
            ["open_scene", "fly_to_pose"].includes(type)
          ),
          true,
          `${query} 的动作类型必须属于稳定协议`,
        );
      }

      assertEquals(
        Array.isArray(data.top_candidates ?? data.candidates ?? []),
        true,
        `${query} 应返回候选数组`,
      );
    }
  },
  sanitizeResources: false,
  sanitizeOps: false,
});
