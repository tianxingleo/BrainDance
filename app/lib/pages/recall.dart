import 'dart:async';
import 'dart:ui' as ui;
import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:markdown/markdown.dart' as md;
import 'package:llamadart/llamadart.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:flutter_highlight/flutter_highlight.dart';
import 'package:flutter_highlight/themes/atom-one-dark.dart';
import 'package:flutter_highlight/themes/atom-one-light.dart';
import '../configs/app_config.dart';
import '../configs/supabase_config.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../extra_func/dir_and_file.dart';
import '../main.dart'
    show
        overviewStatsProvider,
        overviewLocalIndexingProvider,
        pageIndexProvider,
        pendingSubmitTitleProvider,
        recallScrollToTopSignal;
import '../configs/motion_tokens.dart';
import '../services/agent_recall_service.dart';
import '../services/local_rag_index.dart';
import '../services/local_model_catalog_service.dart';
import '../services/local_model_scanner.dart';
import '../services/download_event_bus.dart';
import '../services/viewer_navigation.dart';
import '../widgets/bd_surfaces.dart';
import '../widgets/app_toast.dart';
import 'community/composer_sheet.dart';
import 'community/models.dart';
import 'community/repository.dart';
import 'recall/empty_states.dart';
import 'recall/model_action_overlay.dart';
import 'recall/model_grid.dart';
import 'recall/model_detail_sheet.dart';
import 'recall/time_peeling.dart';
import 'recall/processing_section.dart';
import 'recall/rename_model_dialog.dart';
import 'recall/search_header_section.dart';
import 'recall/search_mode.dart';
import 'recall/agent_asset_card.dart';

part 'recall/recall_agent_runtime.dart';
part 'recall/recall_local_ai.dart';
part 'recall/recall_data_sync.dart';
part 'recall/recall_view.dart';
part 'recall/recall_search.dart';
part 'recall/recall_model_actions.dart';
part 'recall/recall_private_widgets.dart';

class RecallPage extends ConsumerStatefulWidget {
  const RecallPage({super.key});

  @override
  ConsumerState<RecallPage> createState() => _RecallPageState();
}

class _RecallPageState extends ConsumerState<RecallPage> {
  static const String _defaultModelFileName =
      'qwen3-1.7b-braindance-q5-k-m-imatrix.gguf';
  static const String _localModelPathPrefKey = 'recall.local_llm_model_path';
  static const String _localModelUrlPrefKey = 'recall.local_llm_model_url';

  final LocalModelCatalogService _localModelCatalogService =
      const LocalModelCatalogService();
  final TextEditingController _searchController = TextEditingController();
  final TextEditingController _localModelPathController =
      TextEditingController();
  late final TextEditingController _localModelUrlController;
  final LocalRagIndexService _localRagIndex = LocalRagIndexService();
  final Map<String, String> _downloadedLocalModelPathsByUrl = {};
  final Map<String, GlobalKey> _modelCardKeys = {};
  final Map<String, _RecallSearchCacheEntry> _searchCache = {};
  final GlobalKey _actionOverlayStackKey = GlobalKey();
  final GlobalKey<RecallModelActionOverlayState> _overlayKey = GlobalKey();
  final ScrollController _recallScrollController = ScrollController();
  final Set<String> _expandedTaskLogs = {};

  List<Map<String, dynamic>> _models = [];
  List<Map<String, dynamic>> _allModels = [];
  List<Map<String, dynamic>> _processingTasks = [];
  List<LocalModelCatalogItem> _localModelCatalog = const [];
  Map<String, List<String>> _taskAllLogs = {};
  Map<String, dynamic>? _activeModelAction;
  Rect? _activeModelActionRect;
  LocalRagIndexStats? _indexStats;
  RecallSearchMode _searchMode = RecallSearchMode.local;
  LlamaEngine? _localQnaModel;
  StreamSubscription<dynamic>? _llamaStreamSubscription;
  StreamSubscription<String>? _agentStreamSubscription;
  Timer? _modelPollingTimer;
  Timer? _agentElapsedTimer;
  Timer? _agentBootstrapTimer;
  String? _selectedLocalModelUrl;
  String? _activeLocalModelUrl;
  RealtimeChannel? _realtimeChannel;
  String _localAnswer = '';
  String _localReasoning = '';
  String _localAnswerStatus = '端侧模型未加载';
  String _localContextPreview = '';
  String _lastOwnModelSignature = '';
  String? _lastSearchKey;
  String? _agentSessionId;
  String? _agentConversationSummary;
  String? _agentLatestSubmittedQuery;
  AgentSessionState? _agentSessionState;
  final List<AgentConversationEntry> _agentConversationHistory = [];

  ChatMessage? get _agentChatMessage => _agentConversationHistory.isNotEmpty
      ? _agentConversationHistory.first.agentMessage
      : null;

  AgentRecallResponse? get _agentResult => _agentConversationHistory.isNotEmpty
      ? _agentConversationHistory.first.agentResult
      : null;

  set _agentResult(AgentRecallResponse? value) {
    if (_agentConversationHistory.isNotEmpty) {
      _agentConversationHistory.first.agentResult = value;
    }
  }

  AgentStep? _agentBootstrapStep;
  DateTime? _agentRunStartedAt;
  DateTime? _agentRunFinishedAt;
  DateTime? _agentFirstRemoteEventAt;
  bool _isLoading = true;
  bool _isOpeningViewer = false;
  String? _openingViewerLabel;
  bool _isLocalIndexing = false;
  bool _isProcessingExpanded = false;
  bool _isLocalModelLoading = false;
  bool _isLocalModelReady = false;
  bool _isModelDownloading = false;
  bool _didBootstrap = false;
  bool _didFinishInitialModelLoad = false;
  bool _isTabActive = true;
  bool _shouldRefreshProcessingOnResume = false;
  bool _isModelPollingInFlight = false;
  bool _isAgentSearching = false;
  double? _modelDownloadProgress;
  int _modelDownloadedBytes = 0;
  int? _modelDownloadTotalBytes;
  int _searchRequestId = 0;

  final darkBg = const Color(0xFF101014);
  final darkCard = const Color(0xFF18181C);
  final darkInput = const Color(0xFF23232A);
  final darkBorder = const Color(0xFF23232A);

  void _refreshState([VoidCallback? fn]) {
    if (!mounted) {
      return;
    }
    setState(fn ?? () {});
  }

  @override
  void initState() {
    super.initState();
    _initRecallPageState();
  }

  @override
  void dispose() {
    _disposeRecallPageState();
    super.dispose();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _handleRecallPageDependenciesChanged();
  }

  @override
  Widget build(BuildContext context) => _buildRecallPage(context);
}
