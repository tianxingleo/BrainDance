import 'package:flutter/foundation.dart';
import 'package:braindance/configs/supabase_config.dart';

class LocalAiPresetData {
  static bool? _overriddenEnabled;

  static bool get isEnabled =>
      _overriddenEnabled ?? (kDebugMode || SupabaseConfig.isAdminMode);

  static set isEnabled(bool value) {
    _overriddenEnabled = value;
  }

  static void resetToDefault() {
    _overriddenEnabled = null;
  }

  final String question;
  final List<String> keywords;
  final String answer;
  final String reasoning;
  final String contextPreview;

  const LocalAiPresetData({
    required this.question,
    required this.keywords,
    required this.answer,
    required this.reasoning,
    required this.contextPreview,
  });

  static final List<LocalAiPresetData> _presets = [
    const LocalAiPresetData(
      question: '我有哪些模型',
      keywords: ['模型', '有哪些', '列表', '所有', '全部'],
      reasoning: '用户询问模型清单。检索到 3 条记录，涵盖书房、客厅、厨房三个场景。',
      answer:
          '你目前有 3 个模型记录：\n\n'
          '1. **书房场景** — 包含书架、办公桌、显示器（24寸）、笔记本电脑（MacBook Pro）、机械键盘、白板\n'
          '2. **客厅场景** — 包含沙发、茶几、电视柜、落地灯、装饰画\n'
          '3. **厨房场景** — 包含灶台、冰箱、微波炉、烤箱、餐具柜\n\n'
          '如果需要查看某个场景的详细 3D 模型，可以直接说出场景名称。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "书房全景",\n'
          '      "tags": "学习、工作、办公",\n'
          '      "summary": "含办公桌、显示器、笔记本电脑、书架"\n'
          '    },\n'
          '    {\n'
          '      "object": "客厅全景",\n'
          '      "tags": "休闲、会客、娱乐",\n'
          '      "summary": "含沙发、茶几、电视柜、落地灯"\n'
          '    },\n'
          '    {\n'
          '      "object": "厨房全景",\n'
          '      "tags": "烹饪、餐饮、储物",\n'
          '      "summary": "含灶台、冰箱、微波炉、烤箱"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 3,\n'
          '  "intent": "inventory",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '最近添加的模型是什么',
      keywords: ['最近', '最新', '新模型', '最近添加', '新添加'],
      reasoning: '按时间倒序检索，最新记录为书房场景，创建于 2026-05-25。',
      answer:
          '你最近添加的模型是 **「书房场景」**，创建时间是 2026 年 5 月 25 日。\n\n'
          '该场景包含以下物体：\n'
          '- 办公桌（居中位置）\n'
          '- 24寸显示器（办公桌左侧）\n'
          '- MacBook Pro（显示器右侧）\n'
          '- 机械键盘（笔记本前方）\n'
          '- 白板（右侧墙面）\n'
          '- 书架（左侧墙面）\n\n'
          '场景坐标：空间中心点位于 (2.5, 1.8, 0.3)',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "书房全景",\n'
          '      "tags": "学习、工作、办公",\n'
          '      "summary": "含办公桌、显示器、笔记本电脑、书架、白板",\n'
          '      "created_at": "2026-05-25"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "latest_model",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '帮我找一下办公桌',
      keywords: ['办公桌', '桌子', '书桌', '找到', '找'],
      reasoning: '关键词"办公桌"在书房场景的标签和物体描述中命中，相似度 0.87。',
      answer:
          '找到 1 个匹配结果：\n\n'
          '**办公桌** 位于「书房场景」中\n'
          '- 坐标：场景空间 (1.2, 0.0, -0.8)\n'
          '- 标签：`桌面` `办公用品` `实木`\n'
          '- 关联物体：显示器（放置于桌面）、笔记本电脑（桌面右侧）\n\n'
          '所属场景主模型：书房全景（创建于 2026-05-20）\n\n'
          '需要我展开办公桌附近的完整布局吗？',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "书房全景",\n'
          '      "tags": "学习、工作、办公",\n'
          '      "summary": "含办公桌、显示器、笔记本电脑、书架",\n'
          '      "similarity": 0.87\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "object_lookup",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '我的房间里有什么',
      keywords: ['房间', '有什么', '物体', '摆放', '布局'],
      reasoning: '场景"书房"的 meta_info 中包含物体清单和空间关系数据。',
      answer:
          '根据记录，你的 **「书房场景」** 包含以下物体：\n\n'
          '| 物体 | 位置 | 标签 |\n'
          '|------|------|------|\n'
          '| 办公桌 | 房间中央 | 桌面、实木、办公 |\n'
          '| 显示器 | 桌面左侧 | 24寸、电子设备 |\n'
          '| 笔记本电脑 | 桌面右侧 | MacBook Pro、工作 |\n'
          '| 机械键盘 | 笔记本前方 | 输入设备 |\n'
          '| 白板 | 右侧墙面 | 书写、备忘 |\n'
          '| 书架 | 左侧墙面 | 书籍、收纳 |\n\n'
          '物体之间的空间关系：显示器放在办公桌上，笔记本电脑在显示器右侧。书架紧邻左侧墙面，与办公桌距离约 1.2 米。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "书房全景",\n'
          '      "tags": "学习、工作、办公",\n'
          '      "summary": "含办公桌、显示器、笔记本电脑、书架、白板、机械键盘",\n'
          '      "similarity": 0.92\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "room_inventory",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '有没有关于学习的模型',
      keywords: ['学习', '教育', '阅读', '书房', '书本', '书'],
      reasoning: '检索到 2 个带有"学习"相关标签的场景。书房场景标签含"学习""工作"，客厅书架含"阅读"。',
      answer:
          '查到 2 个相关模型：\n\n'
          '1. **书房场景** — 标签包含「学习」「工作」「阅读」\n'
          '   - 相关物体：书架（藏书约 30 册）、办公桌、台灯\n'
          '   - 采光良好，适合长时间学习\n\n'
          '2. **客厅书架** — 标签包含「阅读」「休闲」\n'
          '   - 位于客厅东侧墙面\n'
          '   - 旁边配有落地灯和单人沙发\n\n'
          '两个场景都有充足的照明布置，适合不同的学习场景。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "书房全景",\n'
          '      "tags": "学习、工作、阅读",\n'
          '      "summary": "含书架、办公桌、台灯，采光良好"\n'
          '    },\n'
          '    {\n'
          '      "object": "客厅书架",\n'
          '      "tags": "阅读、休闲",\n'
          '      "summary": "落地灯、单人沙发，适合阅读"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 2,\n'
          '  "intent": "tag_search",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '显示器的位置在哪里',
      keywords: ['显示器', '屏幕', '位置', '在哪里', '坐标'],
      reasoning: '从书房场景的物体坐标数据中找到显示器的位置信息。',
      answer:
          '显示器位于 **书房场景** 的办公桌面上，具体位置：\n\n'
          '- 场景坐标：(1.5, 1.1, -0.5)\n'
          '- 相对于办公桌：桌面左侧\n'
          '- 朝向：用户座位方向\n'
          '- 尺寸：24 英寸\n\n'
          '显示器右侧是一台 MacBook Pro，两者相距约 0.3 米。需要我打开这个场景的 3D 视图吗？',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "书房全景",\n'
          '      "tags": "学习、工作、办公",\n'
          '      "summary": "显示器位于办公桌左侧，坐标(1.5, 1.1, -0.5)"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "spatial_query",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '昨天拍了什么',
      keywords: ['昨天', '拍摄', '扫描', '记录', '新增'],
      reasoning: '日期过滤：昨天（2026-05-26）有 1 条新记录。',
      answer:
          '昨天（2026 年 5 月 26 日）你扫描了 **「阳台花园」** 场景。\n\n'
          '记录概要：\n'
          '- 创建时间：2026-05-26 14:32\n'
          '- 物体数量：8 个（花架、花盆 x4、浇水壶、座椅、小圆桌）\n'
          '- 场景大小：约 4.2m × 3.0m\n'
          '- 状态：已完整重建\n\n'
          '这是最近 7 天内唯一新增的场景。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "阳台花园",\n'
          '      "tags": "户外、植物、休闲",\n'
          '      "summary": "创建于 2026-05-26，含花架、花盆、座椅、小圆桌",\n'
          '      "created_at": "2026-05-26"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "time_filter",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '所有场景汇总',
      keywords: ['汇总', '总结', '统计', '概览', '所有', '全部', '一共'],
      reasoning: '统计全部场景记录的物体、标签和创建时间，进行汇总。',
      answer:
          '你共有 **5 个场景**，汇总如下：\n\n'
          '| 场景 | 物体数 | 创建日期 |\n'
          '|------|--------|----------|\n'
          '| 书房 | 6 个 | 2026-05-20 |\n'
          '| 客厅 | 5 个 | 2026-05-22 |\n'
          '| 厨房 | 4 个 | 2026-05-23 |\n'
          '| 阳台花园 | 8 个 | 2026-05-26 |\n'
          '| 卧室 | 3 个 | 2026-05-18 |\n\n'
          '共记录 **26 个物体**，分布在 5 个场景中。最近一周活跃度：新增 1 个场景，更新 2 个场景。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {"object": "书房全景", "object_count": 6, "created": "2026-05-20"},\n'
          '    {"object": "客厅全景", "object_count": 5, "created": "2026-05-22"},\n'
          '    {"object": "厨房全景", "object_count": 4, "created": "2026-05-23"},\n'
          '    {"object": "阳台花园", "object_count": 8, "created": "2026-05-26"},\n'
          '    {"object": "卧室全景", "object_count": 3, "created": "2026-05-18"}\n'
          '  ],\n'
          '  "hit_count": 5,\n'
          '  "intent": "summary",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
  ];

  static LocalAiPresetData? findMatch(String userQuestion) {
    if (!isEnabled) return null;

    final normalized = userQuestion.trim();
    if (normalized.isEmpty) return null;

    final lower = normalized.toLowerCase();

    // Exact match
    for (final preset in _presets) {
      if (preset.question == normalized ||
          preset.question.toLowerCase() == lower) {
        return preset;
      }
    }

    // Keyword overlap — return preset with most keyword hits
    LocalAiPresetData? bestMatch;
    int bestScore = 0;
    for (final preset in _presets) {
      int score = 0;
      for (final kw in preset.keywords) {
        if (lower.contains(kw.toLowerCase())) {
          score++;
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestMatch = preset;
      }
    }

    return bestScore >= 1 ? bestMatch : null;
  }
}
