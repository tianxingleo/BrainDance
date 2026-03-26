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

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
    _wasCompleted = widget.step.isCompleted;

    widget.step.addListener(_handleStepChange);
  }

  @override
  void dispose() {
    widget.step.removeListener(_handleStepChange);
    super.dispose();
  }

  void _handleStepChange() {
    if (widget.step.isCompleted && !_wasCompleted) {
      _wasCompleted = true;
      Future.delayed(const Duration(milliseconds: 500), () {
        if (mounted) {
          _controller.collapse();
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final toolName = widget.step.toolName ?? '未命名工具';
    final isDone = widget.step.isCompleted;

    final panelColor = widget.isDark
        ? const Color(0xFF1C2128)
        : const Color(0xFFF4F8FB);
    final iconBgColor = widget.isDark
        ? const Color(0xFF2B3A4A) // 暗色背景的图标底色
        : const Color(0xFFDFE8F6); // 亮色背景的图标底色
    final textColor = widget.textColor;
    final hintColor = widget.isDark ? Colors.white60 : Colors.black54;

    return Container(
      decoration: BoxDecoration(
        color: panelColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: widget.isDark
              ? Colors.white10
              : Colors.black.withOpacity(0.06),
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
                  Icons.build_circle_rounded,
                  size: 16,
                  color: widget.isDark
                      ? const Color(0xFF81A2C6)
                      : BDDesign.colorMutedBlue,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  '${textLocalize('agent_status_tool_call')}: $toolName',
                  style: TextStyle(
                    fontSize: 14, // 与回答文字大小差不多
                    color: textColor,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              if (!isDone)
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 6),
                  child: SizedBox(
                    width: 14,
                    height: 14,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                )
              else
                Icon(
                  Icons.check_circle_rounded,
                  size: 16,
                  color: const Color(0xFF4CAF50).withOpacity(0.8),
                ),
            ],
          ),
          childrenPadding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: widget.isDark ? const Color(0xFF12171D) : Colors.white,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: widget.isDark
                      ? Colors.white12
                      : Colors.black.withOpacity(0.04),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    textLocalize('agent_status_tool_args'),
                    style: TextStyle(
                      color: hintColor,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 6),
                  HighlightView(
                    widget.step.content.isEmpty ? '{}' : widget.step.content,
                    language: 'json',
                    theme: widget.isDark ? atomOneDarkTheme : atomOneLightTheme,
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
