import 'package:flutter/foundation.dart';
import 'package:braindance/configs/supabase_config.dart';

class LocalAiPresetData {
  static bool? _overriddenEnabled;

  static bool get isEnabled => _overriddenEnabled ?? true;

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
      reasoning: '用户询问模型清单。检索到 2 条记录：龙王寺户外宗教庭院与现代家居产品展厅。',
      answer:
          '你目前有 2 个模型记录：\n\n'
              '1. **龙王寺户外宗教庭院** — 民间小型佛教活动场所，含汉白玉观音立像、青铜香炉、黄铜大钟、临时板房、盆栽植物、钟亭等 12 个物体\n'
              '2. **现代家居产品展厅** — 生活方式陈列区，含玻璃餐桌、灰蓝陶瓷餐具、编织靠背餐椅、开放式置物柜、原木书架、吊灯、抱枕组合等 19 个物体\n\n'
              '需要查看某个场景的详细信息，可以直接说出场景名称。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "龙王寺户外宗教庭院",\n'
          '      "tags": "户外庭院、佛教场所、观音像、香炉、钟亭",\n'
          '      "summary": "含观音石像、青铜香炉、黄铜大钟、板房、盆栽等 12 个物体"\n'
          '    },\n'
          '    {\n'
          '      "object": "现代家居产品展厅",\n'
          '      "tags": "室内展厅、家居陈列、玻璃餐桌、编织椅、吊灯",\n'
          '      "summary": "含玻璃餐桌、餐具套装、置物柜、书架、展架等 19 个物体"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 2,\n'
          '  "intent": "inventory",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '最近添加的模型是什么',
      keywords: ['最近', '最新', '新模型', '最近添加', '新添加'],
      reasoning: '按时间倒序检索，最新记录为现代家居产品展厅，创建于 2026-05-26。',
      answer:
          '你最近添加的模型是 **「现代家居产品展厅」**，创建时间是 2026 年 5 月 26 日。\n\n'
              '该场景为室内家居产品陈列区，包含以下物体：\n'
              '- 长方形透明玻璃餐桌（前景中央，配 4 把餐椅围合）\n'
              '- 灰蓝色陶瓷餐具套装（碗盘叠放于桌面）\n'
              '- 青绿色陶瓷花瓶 + 干枝尤加利叶（桌面中央装饰）\n'
              '- 双开开放式置物柜（背景主体，黑框白内四层陈列）\n'
              '- 原木色阶梯书架（左侧，顶层金色花盆绿植）\n'
              '- 白色百叶阶梯展架（右侧，展示 3 把中古风餐椅）\n'
              '- 乳白色圆盘吊灯（悬挂于餐桌正上方，暖白光 3000K）\n\n'
              '场景风格：北欧与中古融合，色调以灰蓝、米白、原木棕为主。多视角平视拍摄，无死角盲区。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "现代家居产品展厅",\n'
          '      "tags": "室内展厅、家居陈列、北欧中古融合",\n'
          '      "summary": "含玻璃餐桌、餐具套装、置物柜、书架、展架、吊灯等 19 个物体",\n'
          '      "created_at": "2026-05-26"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "latest_model",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '帮我找一下观音像',
      keywords: ['观音', '观音像', '佛像', '石像', '找', '在哪里'],
      reasoning: '关键词"观音像"在龙王寺户外宗教庭院场景的物体列表和标签中命中，相似度 0.91。',
      answer:
          '找到 1 个匹配结果：\n\n'
              '**白色观音石像** 位于「龙王寺户外宗教庭院」场景中\n'
              '- 材质：白色石材（疑似汉白玉或仿石混凝土），表面光滑带细微雕刻纹理\n'
              '- 高度：约 4–5 米\n'
              '- 特征：手持净瓶与杨柳枝，面容宁静，衣纹流畅，基座为莲花瓣造型\n'
              '- 位置：庭院中后部，居青铜香炉正后方约 1.5 米处，高于香炉约 2 米\n'
              '- 空间关系：与前方香炉、隐含入口构成中轴对称主轴\n\n'
              '所属场景：龙王寺户外宗教庭院（创建于 2026-05-24）\n\n'
              '需要我展开观音像周围的完整布局吗？',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "龙王寺户外宗教庭院",\n'
          '      "tags": "户外庭院、佛教场所、白色观音像、中轴对称",\n'
          '      "summary": "含汉白玉观音立像（4-5米）、青铜香炉、黄铜大钟、板房、盆栽等",\n'
          '      "similarity": 0.91\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "object_lookup",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '这个场景里有什么',
      keywords: ['场景', '有什么', '物体', '摆放', '布局', '里面'],
      reasoning: '从龙王寺户外宗教庭院场景的 meta_info 中提取物体清单与空间关系数据。',
      answer:
          '「龙王寺户外宗教庭院」包含以下物体：\n\n'
          '| 物体 | 位置 | 特征 |\n'
          '|------|------|------|\n'
          '| 白色观音石像 | 庭院中后部 | 汉白玉材质，约4-5米高，手持净瓶柳枝 |\n'
          '| 青铜龙王寺香炉 | 观音像正前方1.5m | 狮首四足，歇山顶，铭文"龍王寺" |\n'
          '| 黄铜大钟 | 右后侧钟亭内 | 表面锤击纹与阴刻铭文 |\n'
          '| 临时板房（左/右） | 庭院两侧 | 彩钢板外墙，左侧前设黄色遮阳棚 |\n'
          '| 盆栽三角梅与绿植 | 香炉两侧及后方 | 紫红色花，陶/瓷/塑料盆 |\n'
          '| 石雕释迦牟尼坐像 | 香炉左后方 | 部分被盆栽遮挡 |\n'
          '| 停车指示牌 | 钟亭前方 | 白底黑字 |\n\n'
          '空间布局：中轴对称式，观音像—香炉—入口构成视觉主轴。地面为雨后湿润水泥地，存在镜面反射。整体氛围肃穆而生活化。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "龙王寺户外宗教庭院",\n'
          '      "tags": "户外庭院、佛教场所、中轴对称、雨后地面",\n'
          '      "summary": "含观音石像、青铜香炉、黄铜大钟、板房、盆栽、钟亭、灭火器等 12 个物体",\n'
          '      "similarity": 0.94\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "scene_inventory",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '有没有家居相关的模型',
      keywords: ['家居', '展厅', '家具', '室内', '陈列', '餐桌', '北欧'],
      reasoning: '检索到 1 条带有"家居""展厅""室内"相关标签的场景记录。',
      answer:
          '查到 1 个相关模型：\n\n'
              '**现代家居产品展厅** — 标签包含「室内展厅」「家居陈列」「北欧中古融合」\n\n'
              '场景概要：\n'
              '- 核心：长方形透明玻璃餐桌 + 4 把风格统一的餐椅围合\n'
              '- 桌面陈设：灰蓝色陶瓷碗盘、琥珀色玻璃杯、黑色烛台与白蜡烛、金属餐具\n'
              '- 背景：大型双开开放式置物柜，四层陈列抱枕、毛毯、陶瓷盘与陶瓶\n'
              '- 左侧：原木色阶梯书架，顶层金色花盆绿植（疑似虎尾兰）\n'
              '- 右侧：白色百叶阶梯展架，展示 3 把中古风木质餐椅\n'
              '- 照明：顶部乳白色圆盘吊灯，暖白光约 3000K\n'
              '- 地面：浅橡木色人字拼地板，局部铺米灰几何纹地毯\n\n'
              '风格关键词：简约、温暖、北欧 × 中古。适合产品陈列参考与室内设计灵感。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "现代家居产品展厅",\n'
          '      "tags": "室内展厅、家居陈列、北欧中古融合、暖白光照明",\n'
          '      "summary": "含玻璃餐桌、餐具套装、置物柜、书架、展架、吊灯、抱枕组合等 19 个物体"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "tag_search",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '香炉的位置在哪里',
      keywords: ['香炉', '青铜', '龙王寺', '位置', '在哪里', '坐标'],
      reasoning: '从龙王寺户外宗教庭院场景的物体关系数据中提取香炉的空间位置信息。',
      answer:
          '青铜香炉位于「龙王寺户外宗教庭院」场景中，具体信息：\n\n'
              '**龙王寺青铜香炉**\n'
              '- 位置：庭院中心线，观音像正前方约 1.5 米处\n'
              '- 材质：青铜铸造，表面氧化呈深灰绿色，局部鎏金\n'
              '- 结构：顶部为中式歇山顶，饰有金色龙形脊饰与悬铃，四足为狮首造型\n'
              '- 铭文：炉身铸有"龍王寺"三字（从右至左读）\n'
              '- 空间关系：与后方观音像、前方隐含入口构成中轴对称主轴；左侧约 2 米为黄色遮阳棚及板房；右后方约 5 米为钟亭\n\n'
              '注意：香炉底部铭文及年代信息因拍摄角度与地面水渍反光无法识别。需要我打开这个场景的 3D 视图吗？',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "龙王寺户外宗教庭院",\n'
          '      "tags": "户外庭院、佛教场所、青铜香炉、中轴对称",\n'
          '      "summary": "青铜龙王寺香炉位于观音像正前方1.5m，庭院中心线，狮首四足",\n'
          '      "similarity": 0.88\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "spatial_query",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '昨天拍了什么',
      keywords: ['昨天', '拍摄', '扫描', '记录', '新增', '最近'],
      reasoning: '日期过滤：昨天（2026-05-26）有 1 条新记录，为现代家居产品展厅。',
      answer:
          '昨天（2026 年 5 月 26 日）你扫描了 **「现代家居产品展厅」** 场景。\n\n'
              '记录概要：\n'
              '- 创建时间：2026-05-26 15:20\n'
              '- 场景类型：室内家居产品展厅（生活方式陈列区）\n'
              '- 物体数量：19 个（玻璃餐桌、餐具套装、花瓶、餐椅 ×4、置物柜、矮柜、书架、绿植、画框、摆件、展架、展示椅 ×3、吊灯、烛台）\n'
              '- 空间大小：约 6.5m × 5.0m\n'
              '- 拍摄参数：多视角平视环绕拍摄，正面全景 + 左前45° + 右前45° + 近景桌面特写\n'
              '- 光照条件：顶部吊灯暖白光（3000K）+ 左侧窗户柔和侧光，整体均匀无硬阴影\n'
              '- 状态：已完整重建\n\n'
              '这是最近 7 天内新增的场景，另一个场景「龙王寺户外宗教庭院」创建于 5 月 24 日。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "现代家居产品展厅",\n'
          '      "tags": "室内展厅、家居陈列、北欧中古融合",\n'
          '      "summary": "创建于 2026-05-26，含玻璃餐桌、餐具套装、置物柜、书架、展架、吊灯等 19 个物体",\n'
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
      reasoning: '统计全部场景记录的物体数量、标签和创建时间，进行汇总分析。',
      answer:
          '你共有 **2 个场景**，汇总如下：\n\n'
              '| 场景 | 物体数 | 创建日期 | 类型 |\n'
              '|------|--------|----------|------|\n'
              '| 龙王寺户外宗教庭院 | 12 个 | 2026-05-24 | 户外宗教场所 |\n'
              '| 现代家居产品展厅 | 19 个 | 2026-05-26 | 室内家居陈列 |\n\n'
              '共记录 **31 个物体**，分布在 2 个场景中。\n\n'
              '场景特征对比：\n'
              '- 龙王寺庭院：中轴对称布局，阴天漫射光，雨后地面反射。核心物体为观音像、香炉、大钟\n'
              '- 家居展厅：纵深线性布局，暖白光吊灯主照明。核心物体为玻璃餐桌、置物柜、阶梯展架\n\n'
              '最近一周活跃度：新增 2 个场景，无更新记录。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {"object": "龙王寺户外宗教庭院", "object_count": 12, "created": "2026-05-24", "type": "户外宗教场所"},\n'
          '    {"object": "现代家居产品展厅", "object_count": 19, "created": "2026-05-26", "type": "室内家居陈列"}\n'
          '  ],\n'
          '  "hit_count": 2,\n'
          '  "intent": "summary",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    // ── 时间类 ──
    const LocalAiPresetData(
      question: '5月份拍了哪些场景',
      keywords: ['5月', '五月', '这个月', '月份', '哪些'],
      reasoning: '按创建时间过滤：2026年5月共有 2 条场景记录。',
      answer:
          '2026 年 5 月你一共扫描了 **2 个场景**：\n\n'
              '1. **龙王寺户外宗教庭院** — 5 月 24 日创建，12 个物体\n'
              '   - 民间小型佛教活动场所，核心为汉白玉观音立像与青铜香炉\n'
              '2. **现代家居产品展厅** — 5 月 26 日创建，19 个物体\n'
              '   - 生活方式陈列区，北欧 × 中古融合风格\n\n'
              '两个场景仅相隔 2 天，分别覆盖了户外宗教与室内家居两个完全不同类型。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {"object": "龙王寺户外宗教庭院", "created": "2026-05-24", "object_count": 12},\n'
          '    {"object": "现代家居产品展厅", "created": "2026-05-26", "object_count": 19}\n'
          '  ],\n'
          '  "hit_count": 2,\n'
          '  "intent": "time_filter",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '这两个场景隔了多久拍的',
      keywords: ['隔了', '多久', '间隔', '时间差', '先后'],
      reasoning: '对比两个场景的创建时间：龙王寺 5/24，家居展厅 5/26。',
      answer:
          '两个场景的拍摄间隔很短：\n\n'
              '- **龙王寺户外宗教庭院**：2026 年 5 月 24 日下午\n'
              '- **现代家居产品展厅**：2026 年 5 月 26 日下午\n\n'
              '两者仅相差 **约 48 小时**。从户外宗教场所到室内家居展厅，拍摄场景类型跨度很大，说明你这两天去了完全不同的地方。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {"object": "龙王寺户外宗教庭院", "created": "2026-05-24T14:10:00"},\n'
          '    {"object": "现代家居产品展厅", "created": "2026-05-26T15:20:00"}\n'
          '  ],\n'
          '  "hit_count": 2,\n'
          '  "intent": "time_compare",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '最早拍的是哪个场景',
      keywords: ['最早', '第一个', '最先', '哪个先'],
      reasoning: '按创建时间升序排列，最早的记录为龙王寺户外宗教庭院（2026-05-24）。',
      answer:
          '你最早拍摄的是 **「龙王寺户外宗教庭院」**，创建于 2026 年 5 月 24 日。\n\n'
              '这是你的第一个 3D 记忆场景，记录了雨后一处民间佛教活动场所的庭院全貌。核心画面是庭院中央约 4-5 米高的汉白玉观音立像与其正前方的青铜"龍王寺"香炉，中轴对称布局庄严肃穆，地面水渍的镜面反射又增添了几分生活气息。\n\n'
              '两天后你拍摄了第二个场景「现代家居产品展厅」。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {"object": "龙王寺户外宗教庭院", "created": "2026-05-24", "object_count": 12},\n'
          '    {"object": "现代家居产品展厅", "created": "2026-05-26", "object_count": 19}\n'
          '  ],\n'
          '  "hit_count": 2,\n'
          '  "intent": "temporal_order",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    // ── 记忆类 ──
    const LocalAiPresetData(
      question: '我记得有个寺庙的场景，里面有个大钟',
      keywords: ['记得', '寺庙', '钟', '大钟', '那个', '好像'],
      reasoning: '用户模糊记忆匹配到龙王寺户外宗教庭院场景，该场景右侧钟亭内确有一口大型黄铜钟。',
      answer:
          '是的，你说的是 **「龙王寺户外宗教庭院」** 场景。\n\n'
              '场景右侧确实有一口大型黄铜钟，具体信息：\n'
              '- 位置：庭院右后侧约 5 米处的简易木构钟亭内\n'
              '- 材质：黄铜铸造，表面可见锤击纹与阴刻铭文\n'
              '- 钟体下半部被钟亭立柱与前方绿植部分遮挡\n\n'
              '同一场景中还有：\n'
              '- 庭院中央的白色观音石像（约 4-5 米高）\n'
              '- 观音像前方的青铜"龍王寺"香炉（狮首四足、金色龙形脊饰）\n'
              '- 左右两侧临时板房、黄色遮阳棚、十余盆三角梅与绿植\n\n'
              '创建时间：2026 年 5 月 24 日。需要我打开这个场景帮你重新看看吗？',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "龙王寺户外宗教庭院",\n'
          '      "tags": "户外庭院、佛教场所、黄铜大钟、钟亭",\n'
          '      "summary": "右侧钟亭内悬挂黄铜大钟，表面有锤击纹与阴刻铭文",\n'
          '      "similarity": 0.85\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "memory_recall",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '帮我想想那天在龙王寺还看到了什么',
      keywords: ['想想', '那天', '龙王寺', '还看到', '回忆', '记不起来'],
      reasoning: '用户试图回忆龙王寺场景中除主要物体外的细节。检索到完整物体清单与场景描述。',
      answer:
          '让我帮你仔细回忆那天在龙王寺看到的画面：\n\n'
              '**你最先注意到的应该是庭院中央的观音像**——约四五米高的汉白玉立像，手持净瓶与杨柳枝，站在莲花基座上，衣纹流畅、面容宁静。像前摆着一座很大的青铜香炉，炉身铸着"龍王寺"三个字。那天刚下过雨，地面还是湿的，水泥地上映着倒影。\n\n'
              '**往左右看**：左边是一排临时板房，前面搭了黄色遮阳棚，棚下隐约能看到供奉台和小佛龛；右边板房旁有个简易木亭，里面挂着一口大黄铜钟，钟上刻了字但下半截被柱子和植物挡住了。\n\n'
              '**细节**：香炉左后方还有一个石雕释迦牟尼坐像，被几盆三角梅半遮着；左侧最边上堆了些橙色塑料周转箱，旁边停了辆盖着紫色布的自行车；墙上还贴了红色消防标语"火灾面前莫惊慌"。\n\n'
              '那天是阴天，光线均匀柔和，远处背景里还有一根很高的绿色通信铁塔。整体就是一个安静、朴素、有人情味的民间宗教小院。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "龙王寺户外宗教庭院",\n'
          '      "tags": "户外庭院、佛教场所、雨后地面、中轴对称",\n'
          '      "summary": "含观音像、香炉、大钟、板房、遮阳棚、盆栽、释迦牟尼坐像等 12 个物体，雨后水泥地面镜面反射"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "episodic_recall",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '那个卖家具的地方叫什么来着',
      keywords: ['家具', '卖', '叫什么', '展厅', '那个地方', '来着'],
      reasoning: '用户模糊描述"卖家具的地方"匹配到现代家居产品展厅场景。',
      answer:
          '你说的是 **「现代家居产品展厅」**，创建于 2026 年 5 月 26 日。\n\n'
              '那是一个生活方式陈列区，不是卖场，更像是品牌展厅。给你回忆一下那个空间：\n\n'
              '一张长方形透明玻璃餐桌摆在正中央，上面铺了灰蓝色陶瓷餐具、琥珀色玻璃杯、黑色烛台，中间一个青绿色花瓶里插着干尤加利叶。四把餐椅围着桌子——两把深色框架配米色编织靠背，另外两把是浅灰布艺软包的。\n\n'
              '头顶一盏乳白色大圆盘吊灯，暖黄光打下来很温馨。背景是一整面墙的双开置物柜，黑框白格子，四层摆满了抱枕、毛毯、陶瓶、小绿釉瓶。左边是原木书架，顶层放了一盆金色花盆的虎尾兰。右边是个白色阶梯展架，上面错落摆了三把中古风木质餐椅。\n\n'
              '整体是灰蓝、米白、原木色，北欧和日式中古混在一起，干净又温暖。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "现代家居产品展厅",\n'
          '      "tags": "室内展厅、家居陈列、北欧中古融合、暖白光",\n'
          '      "summary": "玻璃餐桌 + 餐具 + 置物柜 + 书架 + 展架 + 吊灯，19 个物体，6.5×5m",\n'
          '      "similarity": 0.82\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "memory_recall",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    // ── 情感/印象类 ──
    const LocalAiPresetData(
      question: '你觉得哪个场景拍得最好',
      keywords: ['哪个', '最好', '拍得', '质量', '推荐', '喜欢'],
      reasoning: '从场景复杂度、视角覆盖、光照条件和重建质量四个维度对比两个场景。',
      answer:
          '两个场景各有特点，从 3D 重建角度看：\n\n'
              '**「现代家居产品展厅」综合评分更高** ⭐\n'
              '- 视角覆盖：多视角平视 + 近景特写，无死角盲区\n'
              '- 光照：暖白光吊灯 + 左侧柔和侧光，均匀无硬阴影，高光控制得当\n'
              '- 材质丰富度：玻璃、陶瓷、金属、木材、布艺、皮革，材质对比清晰\n'
              '- 物体密度：19 个物体，层次分明\n\n'
              '**「龙王寺庭院」的独特价值在于**：\n'
              '- 空间尺度更大，中轴对称构图有仪式感\n'
              '- 雨后地面反射带来了独特的镜面效果，但同时也削弱了部分低对比度纹理（如香炉铭文）\n'
              '- 阴天漫射光适合户外重建，但缺少特写角度（钟体铭文、香炉细节未能捕获）\n\n'
              '如果你喜欢精致、温暖、有设计感的画面，家居展厅更好。如果你更喜欢有故事感、有氛围的场景，龙王寺更耐看。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "现代家居产品展厅",\n'
          '      "tags": "室内展厅、光照均匀、视角充分、材质丰富",\n'
          '      "summary": "19 物体，多视角无死角，暖白光均匀照明，重建质量高"\n'
          '    },\n'
          '    {\n'
          '      "object": "龙王寺户外宗教庭院",\n'
          '      "tags": "户外庭院、阴天漫射光、雨后反射、中轴对称",\n'
          '      "summary": "12 物体，空间尺度大，氛围感强，部分细节因角度和反光受限"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 2,\n'
          '  "intent": "quality_compare",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '观音像拍得清楚吗',
      keywords: ['观音', '清楚', '清晰', '拍得', '质量', '细节', '模糊'],
      reasoning: '检查观音像在采集数据中的重建质量：多视角环绕拍摄，主视角平视略仰角聚焦，但缺少极近特写。',
      answer:
          '观音像整体重建质量不错，但有几点需要注意：\n\n'
              '**优势**：\n'
              '- 主视角为平视略仰角，聚焦观音像与香炉，像体整体轮廓和衣纹流畅度捕获充分\n'
              '- 阴天漫射光下无硬阴影，白色石材表面的细微雕刻纹理得以保留\n'
              '- 多视角环绕（左偏 45°、右偏 45°、正面微俯）覆盖了像体各个面\n\n'
              '**局限**：\n'
              '- 缺少极近特写角度，像体面部表情和手持净瓶的细节精度有限\n'
              '- 基座后方的小佛龛被主像遮挡，仅见轮廓\n'
              '- 雨后地面水渍反射造成部分倒影，对深度感知有轻微干扰\n\n'
              '如果你需要更高精度的观音像面部或法器细节，建议下次单独近距离环绕拍摄。',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "龙王寺户外宗教庭院",\n'
          '      "tags": "户外庭院、观音像、多视角环绕、阴天漫射光",\n'
          '      "summary": "汉白玉观音立像约4-5m高，主视角平视仰角，缺极近特写。地面水渍反射微弱干扰深度。整体轮廓与衣纹捕获充分"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 1,\n'
          '  "intent": "quality_check",\n'
          '  "answerability_hint": "hit"\n'
          '}',
    ),
    const LocalAiPresetData(
      question: '有什么让我心情平静的场景',
      keywords: ['心情', '平静', '安静', '放松', '舒服', '治愈', '祥和'],
      reasoning: '从场景氛围分析，龙王寺庭院具有宗教场所特有的肃穆与宁静感；家居展厅暖色调也有安定的包裹感。',
      answer:
          '根据你的两个场景，**两个都能让人平静，但气质完全不同**：\n\n'
              '**「龙王寺户外宗教庭院」—— 出世之静**\n'
              '- 白色观音像手持净瓶立于庭院中央，神情宁静，四周只有盆栽环绕\n'
              '- 阴天灰调的天空、雨后地面的反光，整个氛围收敛、克制、肃穆\n'
              '- 偶尔能想象风吹过悬铃的声响，但那口大钟始终沉默\n'
              '- 适合需要放空、沉淀情绪的时候\n\n'
              '**「现代家居产品展厅」—— 入世之暖**\n'
              '- 乳白色吊灯洒下 3000K 暖光，灰蓝与原木色包围着你\n'
              '- 抱枕叠放、毛毯垂坠、陶瓷器皿安静陈列在置物架上\n'
              '- 像是一个被精心布置好的家，让人觉得被温柔对待\n'
              '- 适合需要一个安稳的、可以喘息的日常角落\n\n'
              '一个在天地之间给你留白，一个在四面墙里给你拥抱。你今天是哪一种？',
      contextPreview:
          '{\n'
          '  "evidence": [\n'
          '    {\n'
          '      "object": "龙王寺户外宗教庭院",\n'
          '      "tags": "户外、宗教、肃穆、宁静、阴天",\n'
          '      "summary": "中轴对称庭院，观音像居中，雨后安静，适合沉淀情绪"\n'
          '    },\n'
          '    {\n'
          '      "object": "现代家居产品展厅",\n'
          '      "tags": "室内、暖光、北欧、温馨、日常",\n'
          '      "summary": "暖白光吊灯，灰蓝原木色调，抱枕与毛毯营造包裹感"\n'
          '    }\n'
          '  ],\n'
          '  "hit_count": 2,\n'
          '  "intent": "emotional_query",\n'
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
