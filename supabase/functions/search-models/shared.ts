import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

export type ChatCompletionResponse = {
  choices?: Array<{
    message?: {
      content?: string | null;
    };
  }>;
};

export type EmbeddingResponse = {
  data?: Array<{
    embedding?: number[];
  }>;
};

export type AiClient = {
  chat: {
    completions: {
      create: (
        options: Record<string, unknown>,
      ) => Promise<ChatCompletionResponse>;
    };
  };
  embeddings: {
    create: (options: Record<string, unknown>) => Promise<EmbeddingResponse>;
  };
};

export type SearchResultRow = Record<string, unknown>;

export type SearchModelsResponse = {
  success: true;
  intent: {
    original_query: string;
    parsed_search_text: string;
    filter_start: string | null;
    filter_end: string | null;
  };
  threshold: number;
  results: SearchResultRow[];
};

const DASHSCOPE_API_URL = Deno.env.get("DASHSCOPE_BASE_URL") ??
  "https://dashscope.aliyuncs.com/compatible-mode/v1";
const DASHSCOPE_EMBEDDING_MODEL = Deno.env.get("DASHSCOPE_EMBEDDING_MODEL") ??
  "text-embedding-v2";

export function safeJsonParse(str: string | null): Record<string, unknown> {
  if (!str) {
    return {};
  }

  try {
    return JSON.parse(str);
  } catch {
    console.error("[Search] JSON 解析失败:", str);
    return {};
  }
}

export function normalizeDate(dateStr: string | null): string | null {
  if (!dateStr) {
    return null;
  }

  const regex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;
  return regex.test(dateStr) ? dateStr : null;
}

export function createAiClient(apiKey: string): AiClient {
  return {
    chat: {
      completions: {
        create: async (options: Record<string, unknown>) => {
          const resp = await fetch(`${DASHSCOPE_API_URL}/chat/completions`, {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${apiKey}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify(options),
          });

          if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`DashScope API 错误: ${resp.status} - ${err}`);
          }

          return resp.json();
        },
      },
    },
    embeddings: {
      create: async (options: Record<string, unknown>) => {
        const resp = await fetch(`${DASHSCOPE_API_URL}/embeddings`, {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(options),
        });

        if (!resp.ok) {
          const err = await resp.text();
          throw new Error(`DashScope API 错误: ${resp.status} - ${err}`);
        }

        return resp.json();
      },
    },
  };
}

export function createSupabaseAdminClient() {
  const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
  const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

  if (!supabaseUrl) {
    throw new Error("未配置 SUPABASE_URL");
  }
  if (!supabaseKey) {
    throw new Error("未配置 SUPABASE_SERVICE_ROLE_KEY");
  }

  return createClient(supabaseUrl, supabaseKey);
}

export async function parseQueryIntent(
  aiClient: AiClient,
  userQuery: string,
): Promise<
  { searchText: string; startTime: string | null; endTime: string | null }
> {
  const today = new Date().toISOString().split("T")[0];
  const systemPrompt = `你是搜索助手。当前日期是: ${today}。
用户会输入一句搜索请求，你需要提取：
1. search_text: 真正用于搜索物体的描述（去掉时间词）。
2. start_time: ISO8601 格式的开始时间 (UTC)，如果没有则为 null。
3. end_time: ISO8601 格式的结束时间 (UTC)，如果没有则为 null。

例子1: "找一下上周拍的红色杯子"
输出: {"search_text": "红色杯子", "start_time": "2026-01-13T00:00:00Z", "end_time": "2026-01-19T23:59:59Z"}

例子2: "搜索之前的猫" (无具体时间)
输出: {"search_text": "猫", "start_time": null, "end_time": null}

只返回 JSON，不要其他内容。`;

  try {
    console.log(`[Search] 正在分析用户意图: "${userQuery}"`);
    const resp = await aiClient.chat.completions.create({
      model: "qwen-turbo",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userQuery },
      ],
      response_format: { type: "json_object" },
    });

    const intentStr = resp.choices?.[0]?.message?.content ?? null;
    const intent = safeJsonParse(intentStr);
    const searchText = (intent.search_text as string) || userQuery;
    const startTime = normalizeDate(intent.start_time as string | null);
    const endTime = normalizeDate(intent.end_time as string | null);

    console.log(
      `[Search] 意图解析完成: text="${searchText}", time=${startTime} to ${endTime}`,
    );
    return { searchText, startTime, endTime };
  } catch (e) {
    console.error("[Search] 意图解析失败，回退到原始查询:", e);
    return { searchText: userQuery, startTime: null, endTime: null };
  }
}

export async function getEmbedding(
  aiClient: AiClient,
  text: string,
): Promise<number[] | null> {
  try {
    const resp = await aiClient.embeddings.create({
      input: [text],
      model: DASHSCOPE_EMBEDDING_MODEL,
    });
    const embedding = resp.data?.[0]?.embedding;
    if (!embedding) {
      console.error("[Search] Embedding API 返回空结果");
      return null;
    }

    console.log(`[Search] 向量生成完成: ${embedding.length} 维`);
    return embedding as number[];
  } catch (e) {
    console.error("[Search] 向量生成失败:", e);
    return null;
  }
}

export async function searchModels(
  supabase: any,
  queryEmbedding: number[],
  matchThreshold: number,
  matchCount: number,
  filterStart: string | null,
  filterEnd: string | null,
): Promise<SearchResultRow[]> {
  console.log(
    `[Search] 执行向量搜索: 阈值=${matchThreshold}, 数量=${matchCount}`,
  );

  const { data, error } = await supabase.rpc("match_memory_poses", {
    query_embedding: queryEmbedding,
    match_threshold: matchThreshold,
    match_count: matchCount,
    filter_start: filterStart,
    filter_end: filterEnd,
  } as never) as { data: unknown; error: { message: string } | null };

  if (error) {
    console.error("[Search] RPC 调用错误:", error);
    throw new Error(`数据库查询失败: ${error.message}`);
  }

  const rows = Array.isArray(data) ? data as SearchResultRow[] : [];
  console.log(`[Search] 找到 ${rows.length} 条结果`);
  return rows;
}

export async function runSearchModelsQuery(
  query: string,
  threshold = 0.5,
): Promise<SearchModelsResponse> {
  const apiKey = Deno.env.get("DASHSCOPE_API_KEY");
  if (!apiKey) {
    throw new Error("未配置 DASHSCOPE_API_KEY");
  }

  const matchThreshold = typeof threshold === "number"
    ? Math.max(0, Math.min(1, threshold))
    : 0.5;

  const aiClient = createAiClient(apiKey);
  const supabase = createSupabaseAdminClient();
  const { searchText, startTime, endTime } = await parseQueryIntent(
    aiClient,
    query,
  );
  const queryVector = await getEmbedding(aiClient, searchText);
  if (!queryVector) {
    throw new Error("向量生成失败");
  }

  const results = await searchModels(
    supabase,
    queryVector,
    matchThreshold,
    10,
    startTime,
    endTime,
  );

  return {
    success: true,
    intent: {
      original_query: query,
      parsed_search_text: searchText,
      filter_start: startTime,
      filter_end: endTime,
    },
    threshold: matchThreshold,
    results,
  };
}
