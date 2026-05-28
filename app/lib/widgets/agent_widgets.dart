import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_highlight/flutter_highlight.dart';
import 'package:flutter_highlight/themes/atom-one-dark.dart';
import 'package:flutter_highlight/themes/atom-one-light.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:markdown/markdown.dart' as md;

import '../../configs/app_config.dart';
import '../../configs/motion_tokens.dart';
import '../../services/agent_recall_service.dart';

class AgentProcessPanel extends StatefulWidget {
  const AgentProcessPanel({
    super.key,
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
  State<AgentProcessPanel> createState() => _AgentProcessPanelState();
}

class _AgentProcessPanelState extends State<AgentProcessPanel>
    with SingleTickerProviderStateMixin {
  static const int _maxStatusSteps = 8;
  late final AnimationController _expandController;
  late final CurvedAnimation _expandCurve;

  @override
  void initState() {
    super.initState();
    _expandController = AnimationController(
      vsync: this,
      duration: BDMotion.durationNormal,
      reverseDuration: const Duration(milliseconds: 220),
    );
    _expandCurve = CurvedAnimation(
      parent: _expandController,
      curve: BDMotion.curveEnter,
      reverseCurve: BDMotion.curveExit,
    );
    if (!_shouldCollapse()) {
      _expandController.value = 1.0;
    }
    widget.chatMessage.addListener(_handleMessageChanged);
  }

  @override
  void didUpdateWidget(covariant AgentProcessPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.chatMessage != widget.chatMessage) {
      oldWidget.chatMessage.removeListener(_handleMessageChanged);
      widget.chatMessage.addListener(_handleMessageChanged);
    }
    _syncExpansion();
  }

  @override
  void dispose() {
    widget.chatMessage.removeListener(_handleMessageChanged);
    _expandController.dispose();
    super.dispose();
  }

  void _handleMessageChanged() {
    if (!mounted) return;
    setState(() {});
    _syncExpansion();
  }

  bool _shouldCollapse() {
    final steps = _buildDisplaySteps(widget.chatMessage.steps);
    return widget.chatMessage.isProcessCollapsed &&
        !widget.isSearching &&
        steps.isNotEmpty;
  }

  void _syncExpansion() {
    if (_shouldCollapse()) {
      if (_expandController.status != AnimationStatus.dismissed &&
          _expandController.status != AnimationStatus.reverse) {
        _expandController.reverse();
      }
    } else {
      if (_expandController.status != AnimationStatus.completed &&
          _expandController.status != AnimationStatus.forward) {
        _expandController.forward();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final steps = _buildDisplaySteps(widget.chatMessage.steps);
    final shouldCollapse = _shouldCollapse();
    final summaryLabel =
        steps.isEmpty ? '等待关键步骤' : '查看 ${steps.length} 个关键步骤';
    final panelColor =
        widget.isDark ? const Color(0xFF10161F) : const Color(0xFFF6F9FC);
    final borderColor = widget.isDark
        ? Colors.white.withValues(alpha: 0.08)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.14);

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: panelColor,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: borderColor),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          InkWell(
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
                  Icon(Icons.memory_rounded, size: 15, color: widget.hintColor),
                  const SizedBox(width: 8),
                  Expanded(
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 200),
                      transitionBuilder: (child, animation) => FadeTransition(
                        opacity: animation,
                        child: child,
                      ),
                      child: Text(
                        shouldCollapse ? summaryLabel : '执行过程',
                        key: ValueKey<bool>(shouldCollapse),
                        style: TextStyle(
                          color: widget.textColor,
                          fontSize: 12.5,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.2,
                        ),
                      ),
                    ),
                  ),
                  if (steps.isNotEmpty)
                    RotationTransition(
                      turns: Tween<double>(begin: 0, end: 0.5)
                          .animate(_expandCurve),
                      child: Icon(
                        Icons.expand_more_rounded,
                        size: 18,
                        color: widget.hintColor,
                      ),
                    ),
                ],
              ),
            ),
          ),
          ClipRect(
            child: AnimatedBuilder(
              animation: _expandCurve,
              builder: (context, child) {
                final value = _expandCurve.value.clamp(0.0, 1.0);
                if (value == 0) return const SizedBox.shrink();
                return Align(
                  alignment: Alignment.topCenter,
                  heightFactor: value,
                  child: Opacity(opacity: value, child: child),
                );
              },
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                child: AgentStepTimeline(
                  steps: steps,
                  isDark: widget.isDark,
                  textColor: widget.textColor,
                  hintColor: widget.hintColor,
                  onRetry: widget.onRetry,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  List<AgentStep> _buildDisplaySteps(List<AgentStep> rawSteps) {
    final filtered = <AgentStep>[];
    for (final step in rawSteps) {
      if (step.type == 'thought') continue;
      if (step.type == 'status') {
        final title = step.compactTitle.trim();
        if (title.isEmpty) continue;
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
    if (statusCount <= _maxStatusSteps) return filtered;
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

class AgentStepTimeline extends StatelessWidget {
  const AgentStepTimeline({
    super.key,
    required this.steps,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.onRetry,
  });

  final List<AgentStep> steps;
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    if (steps.isEmpty) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      children: [
        for (int i = 0; i < steps.length; i++) ...[
          if (i > 0) const SizedBox(height: 8),
          _AgentStepEntryAnimator(
            key: ValueKey(identityHashCode(steps[i])),
            child: ListenableBuilder(
              listenable: steps[i],
              builder: (context, _) {
                final step = steps[i];
                switch (step.type) {
                  case 'status':
                    return AgentStatusStepTile(
                      step: step,
                      isDark: isDark,
                      textColor: textColor,
                      hintColor: hintColor,
                    );
                  case 'tool_call':
                    return AgentToolStepTile(
                      step: step,
                      isDark: isDark,
                      textColor: textColor,
                    );
                  case 'thought':
                    return AgentThoughtStepTile(
                      step: step,
                      isDark: isDark,
                      hintColor: hintColor,
                    );
                  case 'error':
                    return AgentErrorStepTile(step: step, onRetry: onRetry);
                  default:
                    return const SizedBox.shrink();
                }
              },
            ),
          ),
        ],
      ],
    );
  }
}

class _AgentStepEntryAnimator extends StatefulWidget {
  const _AgentStepEntryAnimator({super.key, required this.child});

  final Widget child;

  @override
  State<_AgentStepEntryAnimator> createState() =>
      _AgentStepEntryAnimatorState();
}

class _AgentStepEntryAnimatorState extends State<_AgentStepEntryAnimator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _opacity;
  late final Animation<Offset> _offset;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: BDMotion.durationNormal,
    );
    _opacity = CurvedAnimation(parent: _controller, curve: BDMotion.curveEnter);
    _offset = Tween<Offset>(
      begin: const Offset(0, 0.08),
      end: Offset.zero,
    ).animate(
      CurvedAnimation(parent: _controller, curve: BDMotion.curveFluid),
    );
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _opacity,
      child: SlideTransition(position: _offset, child: widget.child),
    );
  }
}

class AgentStatusStepTile extends StatelessWidget {
  const AgentStatusStepTile({
    super.key,
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
    final panelColor =
        isDark ? const Color(0xFF121A24) : const Color(0xFFF8FBFF);
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

class AgentThoughtStepTile extends StatelessWidget {
  const AgentThoughtStepTile({
    super.key,
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

class AgentErrorStepTile extends StatelessWidget {
  const AgentErrorStepTile({
    super.key,
    required this.step,
    required this.onRetry,
  });

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

class AgentToolStepTile extends StatefulWidget {
  const AgentToolStepTile({
    super.key,
    required this.step,
    required this.isDark,
    required this.textColor,
  });

  final AgentStep step;
  final bool isDark;
  final Color textColor;

  @override
  State<AgentToolStepTile> createState() => _AgentToolStepTileState();
}

class _AgentToolStepTileState extends State<AgentToolStepTile>
    with SingleTickerProviderStateMixin {
  late final AnimationController _expand;
  late final CurvedAnimation _expandCurve;
  late final Animation<double> _iconTurns;
  bool _wasCompleted = false;
  bool _didInitialAutoExpand = false;

  @override
  void initState() {
    super.initState();
    _expand = AnimationController(
      vsync: this,
      duration: BDMotion.durationNormal,
      reverseDuration: const Duration(milliseconds: 220),
    );
    _expandCurve = CurvedAnimation(
      parent: _expand,
      curve: BDMotion.curveEnter,
      reverseCurve: BDMotion.curveExit,
    );
    _iconTurns = Tween<double>(begin: 0, end: 0.5).animate(_expandCurve);
    _wasCompleted = widget.step.isCompleted;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!widget.step.isCompleted && mounted) {
        _didInitialAutoExpand = true;
        _expand.forward();
      }
    });
    widget.step.addListener(_handleStepChange);
  }

  @override
  void dispose() {
    widget.step.removeListener(_handleStepChange);
    _expand.dispose();
    super.dispose();
  }

  void _handleStepChange() {
    if (!mounted) return;
    final isCompleted = widget.step.isCompleted;
    if (isCompleted && !_wasCompleted) {
      _wasCompleted = true;
      Future.delayed(const Duration(milliseconds: 450), () {
        if (mounted) _expand.reverse();
      });
      return;
    }
    if (!isCompleted && _wasCompleted) {
      _wasCompleted = false;
      _expand.forward();
      return;
    }
    if (!isCompleted && !_didInitialAutoExpand) {
      _didInitialAutoExpand = true;
      _expand.forward();
    }
  }

  void _toggle() {
    if (_expand.status == AnimationStatus.completed ||
        _expand.status == AnimationStatus.forward) {
      _expand.reverse();
    } else {
      _expand.forward();
    }
  }

  @override
  Widget build(BuildContext context) {
    final toolName = widget.step.toolName ?? '未命名工具';
    final isDone = widget.step.isCompleted;

    final panelColor =
        widget.isDark ? const Color(0xFF0E1319) : const Color(0xFF11161C);
    final iconBgColor =
        widget.isDark ? const Color(0xFF17212C) : const Color(0xFF1F2A35);
    const textColor = Color(0xFFF5F7FA);
    final hintColor = widget.isDark
        ? Colors.white.withValues(alpha: 0.62)
        : Colors.white.withValues(alpha: 0.72);
    final detailPanelColor =
        widget.isDark ? const Color(0xFF151C25) : const Color(0xFF1A222C);
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
      clipBehavior: Clip.antiAlias,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          InkWell(
            onTap: _toggle,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
              child: Row(
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
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          '$titlePrefix $toolName',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
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
                              Color(0xFF9FD3FF)),
                        ),
                      ),
                    )
                  else
                    Padding(
                      padding: const EdgeInsets.only(right: 4),
                      child: Icon(
                        Icons.check_circle_rounded,
                        size: 16,
                        color: const Color(0xFF42B883).withValues(alpha: 0.9),
                      ),
                    ),
                  RotationTransition(
                    turns: _iconTurns,
                    child: Icon(
                      Icons.expand_more_rounded,
                      size: 18,
                      color: hintColor,
                    ),
                  ),
                ],
              ),
            ),
          ),
          ClipRect(
            child: AnimatedBuilder(
              animation: _expandCurve,
              builder: (context, child) {
                final value = _expandCurve.value.clamp(0.0, 1.0);
                if (value == 0) return const SizedBox.shrink();
                return Align(
                  alignment: Alignment.topCenter,
                  heightFactor: value,
                  child: Opacity(opacity: value, child: child),
                );
              },
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                child: Container(
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
                    mainAxisSize: MainAxisSize.min,
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
                        widget.step.content.isEmpty
                            ? '{}'
                            : widget.step.content,
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
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class CodeElementBuilder extends MarkdownElementBuilder {
  final bool isDark;

  CodeElementBuilder(this.isDark);

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
          color: isDark ? Colors.white10 : Colors.black.withValues(alpha: 0.05),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.05)
                  : Colors.black.withValues(alpha: 0.03),
              borderRadius:
                  const BorderRadius.vertical(top: Radius.circular(8)),
            ),
            child: Text(
              language,
              style: TextStyle(
                fontSize: 12,
                color: isDark ? Colors.white70 : Colors.black54,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(12),
            child: HighlightView(
              textContent,
              language: language,
              theme: isDark ? atomOneDarkTheme : atomOneLightTheme,
              padding: EdgeInsets.zero,
              textStyle:
                  const TextStyle(fontFamily: 'monospace', fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}

class AnimatedMarkdownAnswer extends StatefulWidget {
  const AnimatedMarkdownAnswer({
    super.key,
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
  State<AnimatedMarkdownAnswer> createState() => _AnimatedMarkdownAnswerState();
}

class _AnimatedMarkdownAnswerState extends State<AnimatedMarkdownAnswer> {
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
  void didUpdateWidget(covariant AnimatedMarkdownAnswer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.data == widget.data) return;
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
      builders: {'code': CodeElementBuilder(widget.isDark)},
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
