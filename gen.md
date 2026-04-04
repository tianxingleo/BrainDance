  推荐方案
  先做“并行接入”，不要立刻替换现在的 search-models 列表检索。

  原因很简单：你现在已有的列表链路在 app/lib/pages/recall.dart 已经能跑，而 Viewer 入口 app/lib/pages/webgl_viewer.dart 也已经支持这些字段：

  - initialModelUrl
  - posesUrl
  - sceneId
  - initialPose
  - initialPoseId

  这正好能消费 agent-recall 的两个稳定动作：

  - open_scene
  - fly_to_pose

  所以最稳的做法不是“把 Recall 搜索整体改掉”，而是新增一条“Agent 搜索入口”。

  前端调用链
  建议链路是：

  1. Flutter 输入问题
  2. 调用 Supabase.instance.client.functions.invoke('agent-recall', body: {'query': query})
  3. 解析响应中的：
      - answer
      - evidence
      - actions
  4. 把 answer 显示在 Recall 页顶部或一个结果卡片里
  5. 从 actions 中提取：
      - open_scene.ply
      - open_scene.poses
      - fly_to_pose.matrix
      - fly_to_pose.imageName
  6. 跳转到 WebGLViewerPage

  推荐的数据模型
  你可以在 Flutter 新增一个 service，比如：
  app/lib/services/agent_recall_service.dart

  最小模型建议：

  class AgentRecallResponse {
    final String answer;
    final AgentEvidence? evidence;
    final List<AgentAction> actions;

    AgentRecallResponse({
      required this.answer,
      required this.evidence,
      required this.actions,
    });

    factory AgentRecallResponse.fromJson(Map<String, dynamic> json) {
      return AgentRecallResponse(
        answer: json['answer']?.toString() ?? '',
        evidence: json['evidence'] == null
            ? null
            : AgentEvidence.fromJson(
                Map<String, dynamic>.from(json['evidence'] as Map),
              ),
        actions: ((json['actions'] as List?) ?? [])
            .map((item) => AgentAction.fromJson(Map<String, dynamic>.from(item as Map)))
            .toList(),
      );
    }
  }

  class AgentEvidence {
    final String sceneId;
    final double similarity;
    final List<AgentMatchedFrame> matchedFrames;

    AgentEvidence({
      required this.sceneId,
      required this.similarity,
      required this.matchedFrames,
    });

    factory AgentEvidence.fromJson(Map<String, dynamic> json) {
      return AgentEvidence(
        sceneId: json['sceneId']?.toString() ?? '',
        similarity: (json['similarity'] as num?)?.toDouble() ?? 0,
        matchedFrames: ((json['matchedFrames'] as List?) ?? [])
            .map((item) => AgentMatchedFrame.fromJson(Map<String, dynamic>.from(item as Map)))
            .toList(),
      );
    }
  }

  class AgentMatchedFrame {
    final String imageName;
    final double similarity;
    final List<double>? transformMatrix;

    AgentMatchedFrame({
      required this.imageName,
      required this.similarity,
      required this.transformMatrix,
    });

    factory AgentMatchedFrame.fromJson(Map<String, dynamic> json) {
      final raw = json['transformMatrix'];
      return AgentMatchedFrame(
        imageName: json['imageName']?.toString() ?? '',
        similarity: (json['similarity'] as num?)?.toDouble() ?? 0,
        transformMatrix: raw is List
            ? raw.map((e) => (e as num).toDouble()).toList()
            : null,
      );
    }
  }

  class AgentAction {
    final String type;
    final String sceneId;
    final String? modelId;
    final String? ply;
    final String? poses;
    final String? imageName;
    final List<double>? matrix;

    AgentAction({
      required this.type,
      required this.sceneId,
      this.modelId,
      this.ply,
      this.poses,
      this.imageName,
      this.matrix,
    });

    factory AgentAction.fromJson(Map<String, dynamic> json) {
      final rawMatrix = json['matrix'];
      return AgentAction(
        type: json['type']?.toString() ?? '',
        sceneId: json['sceneId']?.toString() ?? '',
        modelId: json['modelId']?.toString(),
        ply: json['ply']?.toString(),
        poses: json['poses']?.toString(),
        imageName: json['imageName']?.toString(),
        matrix: rawMatrix is List
            ? rawMatrix.map((e) => (e as num).toDouble()).toList()
            : null,
      );
    }
  }

  服务层

  class AgentRecallService {
    final SupabaseClient _client = Supabase.instance.client;

    Future<AgentRecallResponse> query(String query) async {
      final response = await _client.functions.invoke(
        'agent-recall',
        body: {'query': query},
      );

      final data = response.data;
      if (data is! Map) {
        throw Exception('agent-recall 返回格式错误');
      }

      if (data['error'] != null) {
        throw Exception(data['error'].toString());
      }

      return AgentRecallResponse.fromJson(Map<String, dynamic>.from(data));
    }
  }

  如何在 Recall 页面接
  你当前 app/lib/pages/recall.dart 里已经有 _searchModelsFromCloud()，我建议不要直接改它，而是新增：

  - _askAgentRecall(String query)
  - _openAgentRecallResult(AgentRecallResponse result)

  例如：

  Future<AgentRecallResponse> _askAgentRecall(String query) async {
    final service = AgentRecallService();
    return await service.query(query);
  }

  然后从 actions 里抽 Viewer 参数：

  void _openAgentRecallResult(AgentRecallResponse result) {
    final openScene = result.actions.where((a) => a.type == 'open_scene').cast<AgentAction?>().firstOrNull;
    final flyToPose = result.actions.where((a) => a.type == 'fly_to_pose').cast<AgentAction?>().firstOrNull;

    if (openScene == null || openScene.ply == null || openScene.ply!.isEmpty) {
      throw Exception('缺少 open_scene.ply，无法打开 Viewer');
    }

    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => WebGLViewerPage(
          initialModelUrl: openScene.ply!,
          posesUrl: openScene.poses,
          sceneId: openScene.sceneId,
          initialPose: flyToPose?.matrix,
          initialPoseId: flyToPose?.imageName,
        ),
      ),
    );
  }

  UI 上怎么放
  我建议先做一个很轻的入口，不要改现有云端搜索主按钮。

  有两个低风险方案：

  1. 在 Recall 页搜索框旁边新增一个 Agent 按钮
     点它才走 agent-recall
  2. 保留现有云端搜索模式，再加一个新模式
     例如：
      - 云端搜索
      - Agent 搜索

  第二种更适合长期，但第一种改动最小。

  为什么这个方案适合你当前仓库
  因为它和现有实现天然对齐：

  - 现有云端搜索：继续返回列表，适合“找一批结果”
  - 新 Agent 搜索：返回答案 + 证据 + 动作，适合“直接带我去看”
  - Viewer 已经支持：
      - 模型 URL
      - poses URL
      - 初始矩阵
      - 初始 pose id

  所以前端不需要先实现什么复杂 Agent runtime，只要把服务层和跳转层补上。

  我建议的落地顺序

  1. 新增 AgentRecallService
  2. 在 Recall 页加一个 Agent 搜索 入口
  3. 先把 answer 显示出来
  4. 再把 open_scene + fly_to_pose 接到 WebGLViewerPage
  5. 最后再考虑是否把现有云端搜索模式整体升级

  验收标准
  Flutter 侧这轮只看一件事： 

  用户输入一句空间问题后，前端能：

  - 显示 answer
  - 成功打开 WebGLViewerPage
  - 把视角飞到 fly_to_pose.matrix 指向的位置

  如果你要，我下一步可以直接在当前 Flutter 仓库里把这套接法落成代码。