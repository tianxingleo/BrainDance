import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

import '../configs/app_config.dart';
import '../configs/app_theme.dart';
import '../configs/motion_tokens.dart';
import '../models/agent_conversation_model.dart';
import '../services/agent_conversation_db.dart';
import '../services/agent_recall_service.dart';
import '../services/viewer_navigation.dart';
import '../widgets/agent_widgets.dart';
import '../widgets/app_toast.dart';
import 'recall/agent_asset_card.dart';

part 'agent_chat/chat_runtime.dart';
part 'agent_chat/chat_view.dart';
part 'agent_chat/conversation_list.dart';

class AgentChatPage extends ConsumerStatefulWidget {
  const AgentChatPage({super.key});

  @override
  ConsumerState<AgentChatPage> createState() => _AgentChatPageState();
}

class _AgentChatPageState extends ConsumerState<AgentChatPage> {
  final TextEditingController _inputController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final AgentConversationDb _db = AgentConversationDb.instance;

  // Current conversation
  AgentConversation? _currentConversation;
  List<AgentMessageRecord> _messages = [];
  List<AgentConversation> _conversationList = [];

  // Agent runtime state
  String? _agentSessionId;
  String? _agentConversationSummary;
  AgentSessionState? _agentSessionState;
  Map<String, dynamic>? _agentShortTermMemory;

  // Active streaming state
  ChatMessage? _activeChatMessage;
  AgentRecallResponse? _activeResult;
  bool _isSearching = false;
  StreamSubscription<String>? _streamSubscription;
  Timer? _elapsedTimer;
  Timer? _bootstrapTimer;
  DateTime? _runStartedAt;
  DateTime? _runFinishedAt;
  DateTime? _firstRemoteEventAt;
  AgentStep? _bootstrapStep;
  final Set<String> _consumedEventKeys = {};

  bool _isLoadingHistory = true;

  // Cache for restored ChatMessage instances (preserves expand/collapse state)
  final Map<int, ChatMessage> _restoredChatMessages = {};

  @override
  void initState() {
    super.initState();
    _loadConversations();
  }

  @override
  void dispose() {
    _inputController.dispose();
    _scrollController.dispose();
    _streamSubscription?.cancel();
    _elapsedTimer?.cancel();
    _bootstrapTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadConversations() async {
    final conversations = await _db.listConversations();
    if (!mounted) return;
    setState(() {
      _conversationList = conversations;
    });
    if (conversations.isNotEmpty) {
      await _loadConversation(conversations.first);
    } else {
      setState(() {
        _isLoadingHistory = false;
      });
      unawaited(_fetchGreeting());
    }
  }

  Future<void> _loadConversation(AgentConversation conv) async {
    setState(() {
      _isLoadingHistory = true;
    });
    final messages = await _db.getMessages(conv.id);
    if (!mounted) return;
    setState(() {
      _currentConversation = conv;
      _messages = messages;
      _agentSessionId = conv.sessionId;
      _agentConversationSummary = conv.conversationSummary;
      _agentSessionState = conv.sessionStateJson != null
          ? AgentSessionState.fromJson(conv.sessionStateJson!)
          : null;
      _agentShortTermMemory = conv.shortTermMemory;
      _activeChatMessage = null;
      _activeResult = null;
      _isSearching = false;
      _restoredChatMessages.clear();
      _isLoadingHistory = false;
    });
    _scrollToBottom();
  }

  Future<void> _createNewConversation() async {
    await _saveCurrentConversationState();
    final id = '${DateTime.now().millisecondsSinceEpoch}_${Random().nextInt(99999)}';
    final conv = await _db.createConversation(id: id);
    if (!mounted) return;
    setState(() {
      _currentConversation = conv;
      _messages = [];
      _agentSessionId = null;
      _agentConversationSummary = null;
      _agentSessionState = null;
      _agentShortTermMemory = null;
      _activeChatMessage = null;
      _activeResult = null;
      _isSearching = false;
      _restoredChatMessages.clear();
      _conversationList.insert(0, conv);
    });
  }

  Future<void> _deleteConversation(String id) async {
    await _db.deleteConversation(id);
    if (!mounted) return;
    setState(() {
      _conversationList.removeWhere((c) => c.id == id);
      if (_currentConversation?.id == id) {
        _currentConversation = null;
        _messages = [];
        _agentSessionId = null;
        _agentConversationSummary = null;
        _agentSessionState = null;
        _agentShortTermMemory = null;
        _activeChatMessage = null;
        _activeResult = null;
      }
    });
    if (_currentConversation == null && _conversationList.isNotEmpty) {
      await _loadConversation(_conversationList.first);
    }
  }

  Future<void> _saveCurrentConversationState() async {
    final conv = _currentConversation;
    if (conv == null) return;
    conv.sessionId = _agentSessionId;
    conv.conversationSummary = _agentConversationSummary;
    conv.sessionStateJson = _agentSessionState?.toJson();
    conv.shortTermMemory = _agentShortTermMemory;
    await _db.updateConversation(conv);
  }

  Future<void> _submitQuery(String query) async {
    final trimmed = query.trim();
    if (trimmed.isEmpty || _isSearching) return;
    _inputController.clear();
    FocusManager.instance.primaryFocus?.unfocus();

    if (_currentConversation == null) {
      await _createNewConversation();
    }

    final conv = _currentConversation!;
    if (conv.title.isEmpty) {
      conv.title = trimmed.length > 30 ? trimmed.substring(0, 30) : trimmed;
      await _db.updateConversation(conv);
      setState(() {});
    }

    final userMsg = AgentMessageRecord(
      conversationId: conv.id,
      isUser: true,
      content: trimmed,
      timestamp: DateTime.now(),
    );
    final msgId = await _db.insertMessage(userMsg);
    setState(() {
      _messages.add(AgentMessageRecord(
        id: msgId,
        conversationId: userMsg.conversationId,
        isUser: true,
        content: trimmed,
        timestamp: userMsg.timestamp,
      ));
    });
    _scrollToBottom();

    await _askAgent(trimmed);
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    return Scaffold(
      backgroundColor: Colors.transparent,
      drawer: _buildConversationDrawer(isDark),
      body: SafeArea(
        bottom: false,
        child: Column(
          children: [
            _buildAppBar(isDark),
            Expanded(child: _buildChatBody(isDark)),
            _buildInputBar(isDark),
          ],
        ),
      ),
    );
  }
}
