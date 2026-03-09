/**
 * BrainDance 语义搜索 Edge Function
 *
 * ================================================================
 * 项目名称: BrainDance - 面向空间计算时代的三维语义记忆引擎
 * 文件名称: search-models/index.ts
 * 功能描述: 提供自然语言搜索 3D 模型资产的能力
 *
 * 核心功能:
 * 1. 意图解析 - 使用 LLM (Qwen3.5-Plus) 从自然语言中提取搜索词和时间范围
 * 2. 向量生成 - 调用 DashScope text-embedding-v2 生成语义向量
 * 3. 向量搜索 - 通过 pgvector 在 Supabase 中执行相似度搜索
 *
 * 运行方式:
 *   supabase functions serve search-models --no-verify-jwt --env-file .env.local
 *
 * 测试命令:
 *   curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/search-models' \
 *     --header 'Content-Type: application/json' \
 *     --data '{"query":"上周拍的红色杯子"}'
 *
 * 依赖说明:
 * - Deno Runtime: https://deno.com/ (Node.js 的替代品，更安全)
 * - Supabase JS: https://supabase.com/docs/library/getting-started (PostgreSQL + Auth + Storage)
 * - DashScope: https://dashscope.console.aliyun.com/ (阿里云百炼，提供 LLM 和 Embedding)
 *
 * 架构说明:
 * - 使用 Deno 标准库 HTTP 服务器 (serve 函数)
 * - 直接使用 fetch 调用 DashScope API (轻量级，无 SDK 依赖)
 * - 使用 Supabase JS 客户端连接数据库 (支持 pgvector)
 *
 * ================================================================
 * 版本信息
 * ================================================================
 * @version 1.0.0
 * @date 2026-01-20
 * @author BrainDance Team
 */

// ============================================================================
// 【模块导入】导入必要的依赖库
// ============================================================================

/**
 * serve: Deno 标准库中的 HTTP 服务器模块
 * 用于创建 HTTP 服务，监听并处理客户端请求
 *
 * 官方文档: https://deno.land/std@0.168.0/http/server.ts
 *
 * 使用示例:
 * serve((req) => {
 *   return new Response("Hello World");
 * });
 */
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

/**
 * createClient: Supabase JavaScript 客户端工厂函数
 * 用于创建与 Supabase 后端服务通信的客户端
 *
 * 功能:
 * - 连接 PostgreSQL 数据库
 * - 处理用户认证 (Auth)
 * - 访问 Storage 存储桶
 * - 调用 RPC 函数
 *
 * 官方文档: https://supabase.com/docs/reference/javascript/initializing
 *
 * @param url - Supabase 项目 URL
 * @param key - Supabase API Key (Anon Key 或 Service Role Key)
 * @returns Supabase 客户端实例
 */
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// ============================================================================
// 【配置常量】全局配置参数
// ============================================================================

/**
 * CORS (跨域资源共享) 响应头配置
 *
 * 为什么要配置 CORS?
 * - 当前 Edge Function 可能被不同域名的前端应用调用
 * - 浏览器出于安全考虑，会阻止跨域请求
 * - 通过设置 CORS 头，允许特定来源的请求访问资源
 *
 * 配置项说明:
 * - Access-Control-Allow-Origin: 允许访问的来源域名，'*' 表示允许任意来源
 * - Access-Control-Allow-Headers: 允许携带的请求头
 *   * authorization - 用户认证令牌
 *   * x-client-info - 客户端信息
 *   * apikey - Supabase API Key
 *   * content-type - 请求内容类型
 *
 * 生产环境建议:
 * - 不要使用 '*'，而是指定具体允许的域名
 * - 例如: 'https://braindance.app' 或 'http://localhost:3000'
 */
const corsHeaders = {
  "Access-Control-Allow-Origin": "*", // 允许所有来源 (开发环境)
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

/**
 * DashScope API 基础 URL
 *
 * DashScope 是阿里云提供的通义千问大模型服务
 * 这里使用 OpenAI 兼容模式的 API 端点
 *
 * 兼容模式说明:
 * - DashScope 提供了与 OpenAI API 相同的接口格式
 * - 可以使用 OpenAI 的调用方式来访问通义千问模型
 * - 只需要修改 baseURL 和 apiKey 即可
 *
 * API 端点说明:
 * - https://dashscope.aliyuncs.com/compatible-mode/v1
 *   /chat/completions - 聊天补全 (用于意图解析)
 *   /embeddings - 文本向量 (用于语义搜索)
 *
 * 官方文档: https://help.aliyun.com/zh/dashscope/
 */
const DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1";

// ============================================================================
// 【辅助函数】工具性函数，用于数据处理和格式化
// ============================================================================

/**
 * 安全解析 JSON 字符串
 *
 * 功能:
 * - 解析 LLM 可能返回的非标准 JSON 字符串
 * - 处理解析失败的情况，避免程序崩溃
 *
 * 背景说明:
 * - LLM 有时返回的 JSON 可能包含额外字符或格式不规范
 * - 直接使用 JSON.parse() 会抛出异常
 * - 这个函数优雅地处理异常情况
 *
 * @param str - 要解析的 JSON 字符串，可能为 null
 * @returns 解析后的对象，如果解析失败返回空对象 {}
 *
 * 示例:
 * safeJsonParse('{"name": "test"}') // 返回 { name: "test" }
 * safeJsonParse(null)              // 返回 {}
 * safeJsonParse("invalid json")    // 返回 {}，并记录错误日志
 */
function safeJsonParse(str: string | null): Record<string, unknown> {
  // 空值检查
  if (!str) {
    return {};
  }

  try {
    // 正常解析 JSON
    return JSON.parse(str);
  } catch {
    // 解析失败时，记录错误并返回空对象
    // 避免程序崩溃，让程序继续执行
    console.error("[Search] JSON 解析失败:", str);
    return {};
  }
}

/**
 * 验证并规范化日期字符串
 *
 * 功能:
 * - 验证 LLM 返回的日期字符串是否为有效的 ISO 8601 格式
 * - 如果格式无效，返回 null
 *
 * 日期格式要求:
 * - 必须为 ISO 8601 格式: YYYY-MM-DDTHH:mm:ssZ
 * - 例如: "2026-01-20T10:30:00Z"
 * - Z 表示 UTC 时区
 *
 * 为什么要验证?
 * - LLM 有时会产生"幻觉"，返回不存在的日期
 * - 直接将错误日期传给数据库会导致查询失败
 *
 * @param dateStr - 日期字符串，可能为 null
 * @returns 有效的日期字符串，或 null (如果格式无效)
 *
 * 示例:
 * normalizeDate("2026-01-20T10:30:00Z") // 返回 "2026-01-20T10:30:00Z"
 * normalizeDate("2026/01/20")           // 返回 null (格式无效)
 * normalizeDate(null)                   // 返回 null
 */
function normalizeDate(dateStr: string | null): string | null {
  // 空值检查
  if (!dateStr) {
    return null;
  }

  // 正则表达式验证 ISO 8601 格式
  // ^\d{4} - 年份: 4位数字
  // -\d{2} - 月份: - + 2位数字
  // -\d{2} - 日期: - + 2位数字
  // T\d{2}: 时间分隔符 T + 2位小时
  // :\d{2} - 分钟: : + 2位数字
  // :\d{2} - 秒钟: : + 2位数字
  // Z$ - UTC 时区标识 Z，字符串结束
  const regex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;

  // 格式正确返回原值，错误返回 null
  return regex.test(dateStr) ? dateStr : null;
}

/**
 * 创建统一格式的错误响应
 *
 * 功能:
 * - 生成标准化的错误响应 JSON
 * - 自动记录错误日志
 * - 自动包含 CORS 头和 JSON Content-Type
 *
 * 响应格式:
 * {
 *   "success": false,
 *   "error": "错误描述信息"
 * }
 *
 * @param message - 错误信息，将显示给客户端
 * @param status - HTTP 状态码，默认 500 (内部服务器错误)
 * @returns Response 对象，可直接返回给客户端
 *
 * 使用示例:
 * errorResponse("参数错误", 400)        // 返回 400 错误
 * errorResponse("数据库连接失败", 500)  // 返回 500 错误
 */
function errorResponse(message: string, status = 500): Response {
  // 记录错误日志，方便排查问题
  console.error(`[Search] 错误: ${message}`);

  // 返回标准化的错误响应
  return new Response(
    JSON.stringify({
      success: false, // 标识请求失败
      error: message, // 错误描述
    }),
    {
      status, // HTTP 状态码
      headers: {
        ...corsHeaders, // 展开 CORS 配置
        "Content-Type": "application/json", // 声明内容类型为 JSON
      },
    },
  );
}

// ============================================================================
// 【核心业务逻辑】实现具体业务功能的函数
// ============================================================================

/**
 * 解析用户查询意图
 *
 * 功能:
 * - 调用 LLM (通义千问 qwen-plus) 分析用户输入
 * - 提取搜索文本 (去掉时间词等干扰)
 * - 提取时间范围 (开始时间和结束时间)
 *
 * 业务背景:
 * - 用户输入: "找一下上周拍的红色杯子"
 * - LLM 分析后提取:
 *   - search_text: "红色杯子" (真正要搜索的内容)
 *   - start_time: "2026-01-13T00:00:00Z" (上周一的开始)
 *   - end_time: "2026-01-19T23:59:59Z" (上周日的结束)
 *
 * 为什么要做意图解析?
 * - 直接搜索"上周拍的红色杯子"效果不好
 * - 需要去掉时间词"上周拍的"
 * - 需要将相对时间"上周"转换为绝对时间范围
 *
 * @param aiClient - DashScope AI 客户端 (OpenAI 兼容接口)
 * @param userQuery - 用户原始输入的搜索查询
 * @returns 解析后的意图对象，包含搜索文本和时间范围
 *
 * 降级策略:
 * - 如果 LLM 调用失败，返回原始查询，不带时间过滤
 * - 这样至少可以进行基本的语义搜索
 */
async function parseQueryIntent(
  aiClient: typeof OpenAI.prototype,
  userQuery: string,
): Promise<
  { searchText: string; startTime: string | null; endTime: string | null }
> {
  // 获取当前日期，用于相对时间计算
  // 例如: "上周" = 当前日期 - 7 天
  const today = new Date().toISOString().split("T")[0];

  /**
   * System Prompt (系统提示词)
   *
   * 这是给 LLM 的指令，定义它的角色和行为规则
   *
   * 提示词设计要点:
   * 1. 明确当前日期 - 让 LLM 能正确计算相对时间
   * 2. 定义输出格式 - JSON 对象，包含固定字段
   * 3. 提供示例 - Few-shot learning，提高准确性
   * 4. 强调只返回 JSON - 避免 LLM 返回额外解释
   *
   * JSON Object 响应格式:
   * - search_text: 搜索目标描述 (去掉时间词)
   * - start_time: 开始时间 (ISO 8601 UTC 格式)，无则 null
   * - end_time: 结束时间 (ISO 8601 UTC 格式)，无则 null
   */
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
    // 记录日志，方便调试
    console.log(`[Search] 正在分析用户意图: "${userQuery}"`);

    /**
     * 调用 DashScope Chat Completion API
     *
     * 参数说明:
     * - model: 使用的模型
     *   * qwen-plus: Qwen3.5 Plus，最新顶级模型，效果媲美 qwen3-max，支持 1M 上下文
     *   * 相比旧版 qwen-plus，推理能力和多模态能力大幅提升
     * - messages: 对话历史
     *   * system: 系统提示词 (定义 LLM 角色)
     *   * user: 用户查询
     * - response_format: 强制要求返回 JSON
     *   * 避免 LLM 返回自然语言解释
     */
    const resp = await aiClient.chat.completions.create({
      model: "qwen-plus",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userQuery },
      ],
      response_format: { type: "json_object" }, // 强制 JSON 格式
    });

    /**
     * 解析 LLM 响应
     *
     * resp.choices[0].message.content 是 LLM 返回的 JSON 字符串
     * 例如: '{"search_text": "红色杯子", "start_time": "2026-01-13..."}'
     */
    const intentStr = resp.choices[0]?.message?.content;
    const intent = safeJsonParse(intentStr);

    /**
     * 提取并验证各个字段
     *
     * 使用 || 运算符提供默认值:
     * - 如果 LLM 返回空或无效值，使用原始查询
     * - 使用 normalizeDate() 验证时间格式
     */
    const searchText = (intent.search_text as string) || userQuery;
    const startTime = normalizeDate(intent.start_time as string | null);
    const endTime = normalizeDate(intent.end_time as string | null);

    // 记录解析结果，方便验证
    console.log(
      `[Search] 意图解析完成: text="${searchText}", time=${startTime} to ${endTime}`,
    );

    // 返回解析结果
    return { searchText, startTime, endTime };
  } catch (e) {
    // LLM 调用失败时的降级策略
    // 记录错误日志
    console.error("[Search] 意图解析失败，回退到原始查询:", e);

    // 返回原始查询，不带时间过滤
    // 这样至少可以进行基本的语义搜索
    return { searchText: userQuery, startTime: null, endTime: null };
  }
}

/**
 * 生成文本向量 (Embedding)
 *
 * 功能:
 * - 将文本转换为 1536 维的数值向量
 * - 向量捕捉文本的语义信息
 * - 相似的文本会产生相似的向量
 *
 * 技术背景:
 * - 使用 DashScope text-embedding-v2 模型
 * - 输出维度: 1536 维 (float32 数组)
 * - 向量空间: 语义相似的文本在空间中距离更近
 *
 * 应用场景:
 * - 语义搜索: 用户查询 → 向量 → 在向量空间中查找相似向量
 * - 文本相似度: 计算两个向量的余弦相似度
 * - 聚类分析: 将相似文本分组
 *
 * @param aiClient - DashScope AI 客户端
 * @param text - 要生成向量的文本
 * @returns 1536 维向量数组，生成失败返回 null
 *
 * 示例:
 * getEmbedding("红色杯子")
 * // 返回: [0.012, -0.034, 0.056, ..., -0.023] (1536 个浮点数)
 */
async function getEmbedding(
  aiClient: typeof OpenAI.prototype,
  text: string,
): Promise<number[] | null> {
  try {
    /**
     * 调用 DashScope Embedding API
     *
     * 参数说明:
     * - input: 输入文本数组，API 接受批量输入
     *   * 虽然这里只输入一个文本，但 API 设计支持批量处理
     * - model: 使用的嵌入模型
     *   * text-embedding-v2: 1536 维，中英文效果好
     */
    const resp = await aiClient.embeddings.create({
      input: [text], // 包装成数组
      model: "text-embedding-v2",
    });

    // 提取向量
    const embedding = resp.data[0]?.embedding;

    // 向量验证
    if (!embedding) {
      console.error("[Search] Embedding API 返回空结果");
      return null;
    }

    // 记录日志: 向量维度 (应该是 1536)
    console.log(`[Search] 向量生成完成: ${embedding.length} 维`);

    // 返回向量
    return embedding as number[];
  } catch (e) {
    // 生成失败时记录错误
    console.error("[Search] 向量生成失败:", e);
    return null;
  }
}

/**
 * 执行向量相似度搜索
 *
 * 功能:
 * - 在 Supabase 数据库中执行 pgvector 相似度搜索
 * - 使用余弦相似度计算向量距离
 * - 可选的时间范围过滤
 *
 * 技术实现:
 * - 调用 PostgreSQL RPC 函数 match_model_assets
 * - 使用 pgvector 的 <=> 运算符 (余弦距离)
 * - 在数据库层面完成高效搜索
 *
 * RPC 函数参数:
 * - query_embedding: 查询向量 (1536 维)
 * - match_threshold: 相似度阈值 (0.0-1.0)，低于此值不返回
 * - match_count: 返回结果数量上限
 * - filter_start: 时间范围起点 (可选)
 * - filter_end: 时间范围终点 (可选)
 *
 * @param supabase - Supabase 客户端实例
 * @param queryEmbedding - 查询向量 (1536 维)
 * @param matchThreshold - 相似度阈值 (0.7 表示 70% 相似度)
 * @param matchCount - 返回结果数量 (10)
 * @param filterStart - 开始时间 (可选)
 * @param filterEnd - 结束时间 (可选)
 * @returns 搜索结果数组
 *
 * 搜索结果字段:
 * - id: 模型资产 UUID
 * - scene_id: 场景标识符
 * - description: AI 生成的场景描述
 * - ply_path: 3D 模型文件路径
 * - created_at: 创建时间
 * - similarity: 相似度分数 (0.0-1.0)
 */
async function searchModels(
  supabase: ReturnType<typeof createClient>,
  queryEmbedding: number[],
  matchThreshold: number,
  matchCount: number,
  filterStart: string | null,
  filterEnd: string | null,
) {
  // 记录搜索参数
  console.log(
    `[Search] 执行向量搜索: 阈值=${matchThreshold}, 数量=${matchCount}`,
  );

  /**
   * 调用 Supabase RPC 函数
   *
   * supabase.rpc() 用于调用 PostgreSQL 的存储过程
   * match_model_assets 是自定义的向量搜索函数
   *
   * 传递参数必须与 RPC 函数定义完全匹配
   */
  const { data, error } = await supabase.rpc("match_model_assets", {
    query_embedding: queryEmbedding,
    match_threshold: matchThreshold,
    match_count: matchCount,
    filter_start: filterStart,
    filter_end: filterEnd,
  });

  // 错误处理
  if (error) {
    console.error("[Search] RPC 调用错误:", error);
    throw new Error(`数据库查询失败: ${error.message}`);
  }

  // 记录结果数量
  console.log(`[Search] 找到 ${data?.length || 0} 条结果`);

  return data;
}

// ============================================================================
// 【主入口】HTTP 请求处理主函数
// ============================================================================

/**
 * HTTP 请求处理主函数
 *
 * 这是 Edge Function 的入口点
 * 每个 HTTP 请求都会调用这个函数
 *
 * 处理流程:
 * 1. 处理 CORS 预检请求 (OPTIONS)
 * 2. 解析和验证请求参数
 * 3. 加载环境变量
 * 4. 初始化 DashScope 和 Supabase 客户端
 * 5. 执行意图解析 → 向量生成 → 数据库搜索
 * 6. 返回格式化响应
 *
 * @param req - HTTP 请求对象
 * @returns HTTP 响应对象
 *
 * Deno.serve 说明:
 * - 是 Deno 标准库提供的 HTTP 服务器
 * - 自动处理 HTTP 协议细节
 * - 支持异步处理函数
 */
serve(async (req: Request) => {
  /**
   * 步骤 1: 处理 CORS 预检请求
   *
   * 浏览器在发送跨域请求前，会先发送 OPTIONS 请求
   * 询问服务器是否允许该跨域请求
   *
   * 预检请求特点:
   * - 方法是 OPTIONS
   * - 不携带请求体
   * - 只检查 CORS 头
   *
   * 处理方式:
   * - 检测到 OPTIONS 请求，直接返回 200
   * - 带上 CORS 头，告诉浏览器允许的来源和方法
   */
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    /**
     * 步骤 2: 解析请求参数
     *
     * 期望的请求格式 (JSON):
     * {
     *   "query": "搜索关键词",
     *   "threshold": 0.5  // 可选，相似度阈值 (0.0-1.0)，默认 0.5
     * }
     */
    const { query, threshold = 0.5 } = await req.json();

    /**
     * 参数验证
     *
     * 为什么要验证?
     * - 防止无效数据进入后续流程
     * - 给出明确的错误提示
     * - 避免 SQL 注入或 API 调用失败
     */

    // 验证 1: 参数是否存在
    if (!query || typeof query !== "string") {
      return errorResponse("缺少或无效的搜索关键词 'query'", 400);
    }

    // 验证 2: 是否为空字符串
    if (query.trim().length === 0) {
      return errorResponse("搜索关键词不能为空", 400);
    }

    // 验证 3: 长度限制 (防止恶意超长请求)
    if (query.length > 500) {
      return errorResponse("搜索关键词过长（最大 500 字符）", 400);
    }

    // 验证 4: threshold 必须在合理范围内
    const matchThreshold = typeof threshold === "number"
      ? Math.max(0, Math.min(1, threshold))
      : 0.5;

    /**
     * 步骤 3: 加载环境变量
     *
     * Deno.env.get() 用于读取系统环境变量
     * 这些变量在启动 Edge Function 时注入
     *
     * 环境变量来源:
     * - 本地开发: --env-file .env.local 参数
     * - 云端: Supabase Dashboard → Settings → Functions → Secrets
     */
    const apiKey = Deno.env.get("DASHSCOPE_API_KEY");
    const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

    // 验证关键配置是否存在
    if (!apiKey) {
      return errorResponse("未配置 DASHSCOPE_API_KEY", 500);
    }

    /**
     * 步骤 4: 初始化客户端
     *
     * 创建两个客户端:
     * 1. aiClient: 用于调用 DashScope LLM API
     * 2. supabase: 用于连接 Supabase 数据库
     */

    /**
     * 创建 DashScope AI 客户端
     *
     * 这里没有使用 OpenAI SDK，而是直接使用 fetch
     *
     * 优点:
     * - 更轻量，不需要加载整个 SDK
     * - 更容易控制请求细节
     * - 更适合 Edge Function 环境
     *
     * 客户端结构 (模拟 OpenAI SDK 接口):
     * - chat.completions.create() - 聊天补全 (意图解析)
     * - embeddings.create() - 向量生成 (语义搜索)
     */
    const aiClient = {
      chat: {
        completions: {
          create: async (options: Record<string, unknown>) => {
            // 发送 HTTP POST 请求到 DashScope Chat API
            const resp = await fetch(`${DASHSCOPE_API_URL}/chat/completions`, {
              method: "POST",
              headers: {
                "Authorization": `Bearer ${apiKey}`,
                "Content-Type": "application/json",
              },
              body: JSON.stringify(options),
            });

            // 检查 HTTP 状态码
            if (!resp.ok) {
              const err = await resp.text();
              throw new Error(`DashScope API 错误: ${resp.status} - ${err}`);
            }

            // 返回 JSON 响应
            return resp.json();
          },
        },
      },
      embeddings: {
        create: async (options: Record<string, unknown>) => {
          // 发送 HTTP POST 请求到 DashScope Embedding API
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
    } as typeof OpenAI.prototype;

    /**
     * 创建 Supabase 客户端
     *
     * 用于连接 PostgreSQL 数据库
     * 使用 Service Role Key 可以绕过 RLS (Row Level Security)
     *
     * 注意:
     * - Service Role Key 拥有完整权限
     * - 本地开发使用，生产环境应使用 Anon Key
     */
    const supabase = createClient(supabaseUrl, supabaseKey);

    /**
     * 步骤 5: 执行核心业务逻辑
     *
     * 执行顺序:
     * 1. parseQueryIntent() - 意图解析
     * 2. getEmbedding() - 向量生成
     * 3. searchModels() - 数据库搜索
     */

    // 5a. 解析用户意图
    const { searchText, startTime, endTime } = await parseQueryIntent(
      aiClient,
      query,
    );

    // 5b. 生成查询向量
    const queryVector = await getEmbedding(aiClient, searchText);
    if (!queryVector) {
      return errorResponse("向量生成失败", 500);
    }

    // 5c. 执行数据库搜索
    // 默认阈值 0.5 表示返回相似度 50% 以上的结果
    // 用户可通过 threshold 参数自定义 (范围 0.0-1.0)
    // 注意: 当前数据库中的向量可能与 DashScope 生成的不完全匹配
    const results = await searchModels(
      supabase,
      queryVector,
      matchThreshold,
      10,
      startTime,
      endTime,
    );

    /**
     * 步骤 6: 返回成功响应
     *
     * 响应格式:
     * {
     *   "success": true,
     *   "intent": {
     *     "original_query": "原始查询",
     *     "parsed_search_text": "解析后的搜索词",
     *     "filter_start": "时间范围起点",
     *     "filter_end": "时间范围终点"
     *   },
     *   "results": [搜索结果数组]
     * }
     */
    return new Response(
      JSON.stringify({
        success: true,
        intent: {
          original_query: query,
          parsed_search_text: searchText,
          filter_start: startTime,
          filter_end: endTime,
        },
        threshold: matchThreshold,
        results: results || [], // 确保返回数组
      }),
      {
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
        },
      },
    );
  } catch (e) {
    /**
     * 全局异常捕获
     *
     * 捕获所有未处理的异常
     * 返回 500 错误和错误信息
     */
    return errorResponse(e instanceof Error ? e.message : String(e), 500);
  }
});

// ============================================================================
// 【启动日志】服务器启动提示
// ============================================================================

/**
 * 服务器启动日志
 *
 * 当 Edge Function 加载完成时输出
 * 用于确认服务已准备就绪
 *
 * 注意:
 * - 这行代码在模块加载时执行
 * - 不是每个请求都执行
 * - 在 Supabase 本地开发环境会显示
 */
console.log("[Search] Edge Function 已初始化完成，等待请求...");
