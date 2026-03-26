part of '../recall.dart';

class _RecallSearchCacheEntry {
  const _RecallSearchCacheEntry({
    required this.createdAt,
    required this.results,
  });

  final DateTime createdAt;
  final List<Map<String, dynamic>> results;
}

class _ParsedLocalModelOutput {
  const _ParsedLocalModelOutput({
    required this.reasoning,
    required this.answer,
  });

  final String reasoning;
  final String answer;
}

class _AgentProcessPanel extends StatefulWidget {
  const _AgentProcessPanel({
    required this.chatMessage,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.isSearching,
    required this.onRetry,
  });

  final ChatMessage chatMessage;
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final bool isSearching;
  final VoidCallback onRetry;

  @override
  State<_AgentProcessPanel> createState() => _AgentProcessPanelState();
}

class _AgentProcessPanelState extends State<_AgentProcessPanel> {
  final ScrollController _scrollController = ScrollController();
  static const int _maxStatusSteps = 8;

  @override
  void initState() {
    super.initState();
    widget.chatMessage.addListener(_handleMessageChanged);
  }

  @override
  void didUpdateWidget(covariant _AgentProcessPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.chatMessage != widget.chatMessage) {
      oldWidget.chatMessage.removeListener(_handleMessageChanged);
      widget.chatMessage.addListener(_handleMessageChanged);
    }
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
  }

  @override
  void dispose() {
    widget.chatMessage.removeListener(_handleMessageChanged);
    _scrollController.dispose();
    super.dispose();
  }

  void _handleMessageChanged() {
    if (!mounted) {
      return;
    }
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
  }

  void _scrollToBottom() {
    if (!_scrollController.hasClients) {
      return;
    }
    _scrollController.animateTo(
      _scrollController.position.maxScrollExtent,
      duration: const Duration(milliseconds: 220),
      curve: Curves.easeOutCubic,
    );
  }

  @override
  Widget build(BuildContext context) {
    final steps = _buildDisplaySteps(widget.chatMessage.steps);
    final shouldCollapse =
        widget.chatMessage.isProcessCollapsed &&
        !widget.isSearching &&
        steps.isNotEmpty;
    final summaryLabel = steps.isEmpty ? '等待关键步骤' : '查看 ${steps.length} 个关键步骤';
    final panelColor = widget.isDark
        ? const Color(0xFF10161F)
        : const Color(0xFFF6F9FC);
    final borderColor = widget.isDark
        ? Colors.white.withValues(alpha: 0.08)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.14);

    return AnimatedSize(
      duration: const Duration(milliseconds: 260),
      curve: Curves.easeOutCubic,
      child: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: panelColor,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: borderColor),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            InkWell(
              borderRadius: const BorderRadius.vertical(
                top: Radius.circular(14),
              ),
              onTap: steps.isEmpty
                  ? null
                  : () {
                      widget.chatMessage.isProcessCollapsed =
                          !widget.chatMessage.isProcessCollapsed;
                    },
              child: Padding(
                padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
                child: Row(
                  children: [
                    Icon(
                      Icons.memory_rounded,
                      size: 15,
                      color: widget.hintColor,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        shouldCollapse ? summaryLabel : '执行过程',
                        style: TextStyle(
                          color: widget.textColor,
                          fontSize: 12.5,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.2,
                        ),
                      ),
                    ),
                    if (steps.isNotEmpty)
                      Icon(
                        shouldCollapse
                            ? Icons.expand_more_rounded
                            : Icons.expand_less_rounded,
                        size: 18,
                        color: widget.hintColor,
                      ),
                  ],
                ),
              ),
            ),
            AnimatedCrossFade(
              firstChild: const SizedBox.shrink(),
              secondChild: Padding(
                padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                child: _AgentStepTimeline(
                  steps: steps,
                  isDark: widget.isDark,
                  textColor: widget.textColor,
                  hintColor: widget.hintColor,
                  scrollController: _scrollController,
                  onRetry: widget.onRetry,
                ),
              ),
              crossFadeState: shouldCollapse
                  ? CrossFadeState.showFirst
                  : CrossFadeState.showSecond,
              duration: const Duration(milliseconds: 220),
              sizeCurve: Curves.easeOutCubic,
            ),
          ],
        ),
      ),
    );
  }

  List<AgentStep> _buildDisplaySteps(List<AgentStep> rawSteps) {
    final filtered = <AgentStep>[];

    for (final step in rawSteps) {
      if (step.type == 'thought') {
        continue;
      }

      if (step.type == 'status') {
        final title = step.compactTitle.trim();
        if (title.isEmpty) {
          continue;
        }
        if (filtered.isNotEmpty) {
          final last = filtered.last;
          if (last.type == 'status' && last.compactTitle.trim() == title) {
            filtered[filtered.length - 1] = step;
            continue;
          }
        }
      }

      filtered.add(step);
    }

    var statusCount = filtered.where((step) => step.type == 'status').length;
    if (statusCount <= _maxStatusSteps) {
      return filtered;
    }

    final trimmed = <AgentStep>[];
    for (final step in filtered.reversed) {
      if (step.type == 'status') {
        if (statusCount > _maxStatusSteps) {
          statusCount -= 1;
          continue;
        }
      }
      trimmed.add(step);
    }
    return trimmed.reversed.toList(growable: false);
  }
}

class _AgentStepTimeline extends StatelessWidget {
  const _AgentStepTimeline({
    required this.steps,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.scrollController,
    required this.onRetry,
  });

  final List<AgentStep> steps;
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final ScrollController scrollController;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final fadeBase = isDark ? const Color(0xFF10161F) : const Color(0xFFF6F9FC);
    return ConstrainedBox(
      constraints: const BoxConstraints(maxHeight: 220),
      child: ShaderMask(
        shaderCallback: (bounds) {
          return LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [fadeBase.withValues(alpha: 0), fadeBase, fadeBase],
            stops: const [0, 0.18, 1],
          ).createShader(bounds);
        },
        blendMode: BlendMode.dstIn,
        child: ListView.separated(
          controller: scrollController,
          padding: EdgeInsets.zero,
          itemCount: steps.length,
          separatorBuilder: (_, _) => const SizedBox(height: 8),
          itemBuilder: (context, index) {
            final step = steps[index];
            return ListenableBuilder(
              listenable: step,
              builder: (context, _) {
                switch (step.type) {
                  case 'status':
                    return _AgentStatusStepTile(
                      step: step,
                      isDark: isDark,
                      textColor: textColor,
                      hintColor: hintColor,
                    );
                  case 'tool_call':
                    return _AgentStepTile(
                      step: step,
                      isDark: isDark,
                      textColor: textColor,
                    );
                  case 'thought':
                    return _AgentThoughtStepTile(
                      step: step,
                      isDark: isDark,
                      hintColor: hintColor,
                    );
                  case 'error':
                    return _AgentErrorStepTile(step: step, onRetry: onRetry);
                  default:
                    return const SizedBox.shrink();
                }
              },
            );
          },
        ),
      ),
    );
  }
}

class _AgentStatusStepTile extends StatelessWidget {
  const _AgentStatusStepTile({
    required this.step,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
  });

  final AgentStep step;
  final bool isDark;
  final Color textColor;
  final Color hintColor;

  @override
  Widget build(BuildContext context) {
    final panelColor = isDark
        ? const Color(0xFF121A24)
        : const Color(0xFFF8FBFF);
    final borderColor = isDark
        ? Colors.white12
        : BDDesign.colorMutedBlue.withValues(alpha: 0.12);
    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: panelColor,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: borderColor),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(top: 1),
            child: Icon(
              step.isCompleted ? Icons.check_rounded : Icons.timelapse_rounded,
              size: 14,
              color: step.isCompleted ? const Color(0xFF42B883) : hintColor,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              step.content,
              style: TextStyle(
                color: step.isCompleted ? hintColor : textColor,
                fontSize: 11.8,
                height: 1.45,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _AgentThoughtStepTile extends StatelessWidget {
  const _AgentThoughtStepTile({
    required this.step,
    required this.isDark,
    required this.hintColor,
  });

  final AgentStep step;
  final bool isDark;
  final Color hintColor;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(top: 2),
            child: Icon(
              Icons.psychology_alt_outlined,
              size: 14,
              color: hintColor.withValues(alpha: 0.8),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              step.content.trim(),
              style: TextStyle(
                color: hintColor,
                fontSize: 11.5,
                height: 1.5,
                fontStyle: FontStyle.italic,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _AgentErrorStepTile extends StatelessWidget {
  const _AgentErrorStepTile({required this.step, required this.onRetry});

  final AgentStep step;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0x1AF44336),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0x66F44336)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.error_outline, color: Colors.red, size: 16),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  step.content,
                  style: const TextStyle(
                    color: Colors.red,
                    fontSize: 12.5,
                    height: 1.4,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.red,
              side: const BorderSide(color: Colors.red, width: 1),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 0),
            ),
            onPressed: onRetry,
            icon: const Icon(Icons.refresh, size: 14),
            label: const Text('重试', style: TextStyle(fontSize: 12)),
          ),
        ],
      ),
    );
  }
}

class _AgentStepTile extends StatefulWidget {
  final AgentStep step;
  final bool isDark;
  final Color textColor;

  const _AgentStepTile({
    required this.step,
    required this.isDark,
    required this.textColor,
  });

  @override
  State<_AgentStepTile> createState() => _AgentStepTileState();
}

class _AgentStepTileState extends State<_AgentStepTile>
    with AutomaticKeepAliveClientMixin {
  final ExpansibleController _controller = ExpansibleController();
  bool _wasCompleted = false;
  bool _didInitialAutoExpand = false;

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
    _wasCompleted = widget.step.isCompleted;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!widget.step.isCompleted && mounted) {
        _didInitialAutoExpand = true;
        _controller.expand();
      }
    });
    widget.step.addListener(_handleStepChange);
  }

  @override
  void dispose() {
    widget.step.removeListener(_handleStepChange);
    super.dispose();
  }

  void _handleStepChange() {
    final isCompleted = widget.step.isCompleted;
    if (isCompleted && !_wasCompleted) {
      _wasCompleted = true;
      Future.delayed(const Duration(milliseconds: 450), () {
        if (mounted) {
          _controller.collapse();
        }
      });
      return;
    }
    if (!isCompleted && _wasCompleted) {
      _wasCompleted = false;
      if (mounted) {
        _controller.expand();
      }
      return;
    }
    if (!isCompleted && !_didInitialAutoExpand && mounted) {
      _didInitialAutoExpand = true;
      _controller.expand();
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final toolName = widget.step.toolName ?? '未命名工具';
    final isDone = widget.step.isCompleted;

    final panelColor = widget.isDark
        ? const Color(0xFF0E1319)
        : const Color(0xFF11161C);
    final iconBgColor = widget.isDark
        ? const Color(0xFF17212C)
        : const Color(0xFF1F2A35);
    final textColor = const Color(0xFFF5F7FA);
    final hintColor = widget.isDark
        ? Colors.white.withValues(alpha: 0.62)
        : Colors.white.withValues(alpha: 0.72);
    final detailPanelColor = widget.isDark
        ? const Color(0xFF151C25)
        : const Color(0xFF1A222C);
    final statusLabel = isDone ? 'Finished' : 'Running';
    final titlePrefix = isDone ? 'Finished' : 'Using';

    return Container(
      decoration: BoxDecoration(
        color: panelColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: widget.isDark
              ? Colors.white10
              : Colors.white.withValues(alpha: 0.08),
        ),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          controller: _controller,
          tilePadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          iconColor: hintColor,
          collapsedIconColor: hintColor,
          title: Row(
            children: [
              Container(
                width: 26,
                height: 26,
                decoration: BoxDecoration(
                  color: iconBgColor,
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Icon(
                  isDone
                      ? Icons.terminal_rounded
                      : Icons.developer_mode_rounded,
                  size: 15,
                  color: const Color(0xFF9FD3FF),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '$titlePrefix $toolName',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 13,
                        color: textColor,
                        fontWeight: FontWeight.w600,
                        fontFamily: 'monospace',
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      statusLabel,
                      style: TextStyle(
                        fontSize: 10.5,
                        color: hintColor,
                        letterSpacing: 0.3,
                      ),
                    ),
                  ],
                ),
              ),
              if (!isDone)
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 6),
                  child: SizedBox(
                    width: 14,
                    height: 14,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        Color(0xFF9FD3FF),
                      ),
                    ),
                  ),
                )
              else
                Icon(
                  Icons.check_circle_rounded,
                  size: 16,
                  color: const Color(0xFF42B883).withValues(alpha: 0.9),
                ),
            ],
          ),
          childrenPadding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: detailPanelColor,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: widget.isDark
                      ? Colors.white12
                      : Colors.white.withValues(alpha: 0.08),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    textLocalize('agent_status_tool_args'),
                    style: TextStyle(
                      color: hintColor,
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      fontFamily: 'monospace',
                    ),
                  ),
                  const SizedBox(height: 6),
                  HighlightView(
                    widget.step.content.isEmpty ? '{}' : widget.step.content,
                    language: 'json',
                    theme: atomOneDarkTheme,
                    padding: EdgeInsets.zero,
                    textStyle: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 12,
                      height: 1.45,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CodeElementBuilder extends MarkdownElementBuilder {
  final bool isDark;
  final BuildContext context;

  _CodeElementBuilder(this.isDark, this.context);

  @override
  Widget? visitElementAfter(md.Element element, TextStyle? preferredStyle) {
    var language = 'plaintext';
    if (element.attributes['class'] != null) {
      String lg = element.attributes['class'] as String;
      if (lg.startsWith('language-')) {
        language = lg.substring(9);
      }
    }
    final textContent = element.textContent.trim();
    if (textContent.isEmpty) return null;

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8),
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF13181E) : const Color(0xFFF1F4EA),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: isDark ? Colors.white10 : Colors.black.withOpacity(0.05),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: isDark
                  ? Colors.white.withOpacity(0.05)
                  : Colors.black.withOpacity(0.03),
              borderRadius: const BorderRadius.vertical(
                top: Radius.circular(8),
              ),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  language,
                  style: TextStyle(
                    fontSize: 12,
                    color: isDark ? Colors.white70 : Colors.black54,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                InkWell(
                  onTap: () {
                    Clipboard.setData(ClipboardData(text: textContent));
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text(textLocalize('agent_action_code_copied')),
                        duration: const Duration(seconds: 2),
                      ),
                    );
                  },
                  child: Row(
                    children: [
                      Icon(
                        Icons.copy,
                        size: 14,
                        color: isDark ? Colors.white70 : Colors.black54,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        textLocalize('agent_action_copy'),
                        style: TextStyle(
                          fontSize: 12,
                          color: isDark ? Colors.white70 : Colors.black54,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(12),
            child: HighlightView(
              textContent,
              language: language,
              theme: isDark ? atomOneDarkTheme : atomOneLightTheme,
              padding: EdgeInsets.zero,
              textStyle: const TextStyle(fontFamily: 'monospace', fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}

class _AnimatedMarkdownAnswer extends StatefulWidget {
  const _AnimatedMarkdownAnswer({
    required this.data,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
  });

  final String data;
  final bool isDark;
  final Color textColor;
  final Color hintColor;

  @override
  State<_AnimatedMarkdownAnswer> createState() =>
      _AnimatedMarkdownAnswerState();
}

class _AnimatedMarkdownAnswerState extends State<_AnimatedMarkdownAnswer> {
  Timer? _timer;
  int _visibleLength = 0;
  String _animatedText = '';

  @override
  void initState() {
    super.initState();
    _animatedText = widget.data;
    _visibleLength = widget.data.length;
  }

  @override
  void didUpdateWidget(covariant _AnimatedMarkdownAnswer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.data == widget.data) {
      return;
    }
    final next = widget.data;
    if (!next.startsWith(_animatedText)) {
      _timer?.cancel();
      _animatedText = next;
      _visibleLength = next.length;
      return;
    }
    _animatedText = next;
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(milliseconds: 14), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      if (_visibleLength >= _animatedText.length) {
        timer.cancel();
        return;
      }
      setState(() {
        _visibleLength = (_visibleLength + 3).clamp(0, _animatedText.length);
      });
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final visibleText = _animatedText.substring(0, _visibleLength);
    return MarkdownBody(
      data: visibleText,
      builders: {'code': _CodeElementBuilder(widget.isDark, context)},
      styleSheet: MarkdownStyleSheet(
        p: TextStyle(color: widget.textColor, fontSize: 14, height: 1.6),
        h1: TextStyle(
          color: widget.textColor,
          fontSize: 18,
          height: 1.3,
          fontWeight: FontWeight.w700,
        ),
        h2: TextStyle(
          color: widget.textColor,
          fontSize: 16,
          height: 1.3,
          fontWeight: FontWeight.w700,
        ),
        blockquote: TextStyle(
          color: widget.hintColor,
          fontSize: 13,
          height: 1.5,
        ),
        code: TextStyle(
          color: widget.textColor,
          fontSize: 12.5,
          fontFamily: 'monospace',
        ),
        codeblockDecoration: BoxDecoration(
          color: widget.isDark
              ? const Color(0xFF131813)
              : const Color(0xFFF1F4EA),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: BDDesign.colorFadedOlive.withValues(
              alpha: widget.isDark ? 0.28 : 0.18,
            ),
          ),
        ),
      ),
    );
  }
}
