part of '../agent_chat.dart';

extension _TimeCompareView on _AgentChatPageState {
  Widget _buildTimeCompareSection({
    required CompareContext data,
    required AgentRecallResponse result,
    required bool isDark,
    required Color textColor,
    required Color hintColor,
  }) {
    final baseline = data.baseline;
    final target = data.target;
    final windows = data.windows;

    return Padding(
      padding: const EdgeInsets.only(top: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (windows != null) ...[
            _TimeCompareWindowsBanner(
              windows: windows,
              isDark: isDark,
              hintColor: hintColor,
              textColor: textColor,
            ),
            const SizedBox(height: 12),
          ],
          LayoutBuilder(
            builder: (context, constraints) {
              final stacked = constraints.maxWidth < 560;
              if (stacked) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    if (baseline != null)
                      _TimeCompareSceneCard(
                        evidence: baseline,
                        slotLabel: '旧版基线',
                        accent: const Color(0xFF8C7B5B),
                        isDark: isDark,
                        textColor: textColor,
                        hintColor: hintColor,
                        onOpen: _buildCompareOnOpen(
                          slot: AgentActionSlot.baseline,
                          evidence: baseline,
                          actions: result.actions,
                        ),
                      ),
                    if (baseline != null && target != null)
                      const SizedBox(height: 12),
                    if (target != null)
                      _TimeCompareSceneCard(
                        evidence: target,
                        slotLabel: '新版目标',
                        accent: BDDesign.colorMutedBlue,
                        isDark: isDark,
                        textColor: textColor,
                        hintColor: hintColor,
                        onOpen: _buildCompareOnOpen(
                          slot: AgentActionSlot.target,
                          evidence: target,
                          actions: result.actions,
                        ),
                      ),
                  ],
                );
              }
              return Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (baseline != null)
                    Expanded(
                      child: _TimeCompareSceneCard(
                        evidence: baseline,
                        slotLabel: '旧版基线',
                        accent: const Color(0xFF8C7B5B),
                        isDark: isDark,
                        textColor: textColor,
                        hintColor: hintColor,
                        onOpen: _buildCompareOnOpen(
                          slot: AgentActionSlot.baseline,
                          evidence: baseline,
                          actions: result.actions,
                        ),
                      ),
                    ),
                  if (baseline != null && target != null)
                    const SizedBox(width: 12),
                  if (target != null)
                    Expanded(
                      child: _TimeCompareSceneCard(
                        evidence: target,
                        slotLabel: '新版目标',
                        accent: BDDesign.colorMutedBlue,
                        isDark: isDark,
                        textColor: textColor,
                        hintColor: hintColor,
                        onOpen: _buildCompareOnOpen(
                          slot: AgentActionSlot.target,
                          evidence: target,
                          actions: result.actions,
                        ),
                      ),
                    ),
                ],
              );
            },
          ),
          if (_hasAnyDiff(data.diff)) ...[
            const SizedBox(height: 12),
            _TimeCompareDiffPanel(
              diff: data.diff,
              isDark: isDark,
              textColor: textColor,
              hintColor: hintColor,
            ),
          ],
        ],
      ),
    );
  }

  bool _hasAnyDiff(CompareDiff diff) =>
      diff.commonObjects.isNotEmpty ||
      diff.addedObjects.isNotEmpty ||
      diff.removedObjects.isNotEmpty ||
      diff.commonTags.isNotEmpty ||
      diff.addedTags.isNotEmpty ||
      diff.removedTags.isNotEmpty ||
      diff.limitations.isNotEmpty;

  VoidCallback? _buildCompareOnOpen({
    required AgentActionSlot slot,
    required CompareSceneEvidence evidence,
    required List<AgentAction> actions,
  }) {
    final openAction = actions
        .where((a) => a.type == 'open_scene' && a.slot == slot)
        .cast<AgentAction?>()
        .firstWhere((_) => true, orElse: () => null);
    final flyAction = actions
        .where((a) => a.type == 'fly_to_pose' && a.slot == slot)
        .cast<AgentAction?>()
        .firstWhere((_) => true, orElse: () => null);

    final ply = openAction?.ply ?? evidence.ply ?? '';
    final sceneId = openAction?.sceneId.isNotEmpty == true
        ? openAction!.sceneId
        : evidence.sceneId;
    if (ply.isEmpty || sceneId.isEmpty) return null;

    return () {
      final modelUrl = ply.startsWith('http://') || ply.startsWith('https://')
          ? ply
          : toPublicUrl(ply);
      final posesUrl = (openAction?.poses != null &&
              openAction!.poses!.isNotEmpty &&
              !openAction.poses!.startsWith('http'))
          ? toPublicUrl(openAction.poses!)
          : (openAction?.poses ?? evidence.poses ?? toPosesUrl(ply));
      unawaited(
        openViewer(
          context,
          initialModelUrl: modelUrl,
          posesUrl: posesUrl,
          sceneId: sceneId,
          initialPose: flyAction?.matrix,
          initialPoseId: flyAction?.imageName,
        ),
      );
    };
  }
}

class _TimeCompareWindowsBanner extends StatelessWidget {
  final CompareWindows windows;
  final bool isDark;
  final Color textColor;
  final Color hintColor;

  const _TimeCompareWindowsBanner({
    required this.windows,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
  });

  @override
  Widget build(BuildContext context) {
    final focus = windows.compareFocus?.trim();
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: isDark
            ? const Color(0xFF1A1F26)
            : BDDesign.colorMutedBlueLight.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: BDDesign.colorMutedBlue.withValues(alpha: isDark ? 0.5 : 0.3),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.compare_arrows_rounded,
                  size: 16, color: BDDesign.colorMutedBlue),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  '时间对比窗口',
                  style: TextStyle(
                    color: textColor,
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              if (focus != null && focus.isNotEmpty)
                Flexible(
                  child: Text(
                    '焦点：$focus',
                    style: TextStyle(color: hintColor, fontSize: 11.5),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 6),
          _windowLine('旧版', windows.baseline, hintColor),
          const SizedBox(height: 2),
          _windowLine('新版', windows.target, hintColor),
        ],
      ),
    );
  }

  Widget _windowLine(String label, CompareTimeWindow w, Color hintColor) {
    final start = _shortIso(w.startTime);
    final end = _shortIso(w.endTime);
    return Text(
      '$label：$start  →  $end',
      style: TextStyle(color: hintColor, fontSize: 11.5),
    );
  }

  String _shortIso(String iso) {
    if (iso.isEmpty) return '—';
    final t = iso.replaceAll('T', ' ');
    return t.length > 16 ? t.substring(0, 16) : t;
  }
}

class _TimeCompareSceneCard extends StatelessWidget {
  final CompareSceneEvidence evidence;
  final String slotLabel;
  final Color accent;
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final VoidCallback? onOpen;

  const _TimeCompareSceneCard({
    required this.evidence,
    required this.slotLabel,
    required this.accent,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    final desc = evidence.description?.trim() ?? '';
    final title = (evidence.displayName?.trim().isNotEmpty ?? false)
        ? evidence.displayName!.trim()
        : (desc.isNotEmpty ? desc : evidence.sceneId);
    final created = _shortDate(evidence.createdAt);
    final similarityPct =
        '${(evidence.similarity * 100).toStringAsFixed(1)}%';

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isDark
            ? const Color(0xFF161B22)
            : Colors.white.withValues(alpha: 0.92),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: accent.withValues(alpha: isDark ? 0.55 : 0.45),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: accent.withValues(alpha: 0.18),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  slotLabel,
                  style: TextStyle(
                    color: accent,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              const Spacer(),
              Text(
                similarityPct,
                style: TextStyle(color: hintColor, fontSize: 11.5),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            title,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              color: textColor,
              fontSize: 14,
              fontWeight: FontWeight.w600,
              height: 1.3,
            ),
          ),
          if (desc.isNotEmpty && desc != title) ...[
            const SizedBox(height: 4),
            Text(
              desc,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: hintColor, fontSize: 12, height: 1.4),
            ),
          ],
          if (created.isNotEmpty) ...[
            const SizedBox(height: 6),
            Row(
              children: [
                Icon(Icons.event_outlined, size: 13, color: hintColor),
                const SizedBox(width: 4),
                Text(created,
                    style: TextStyle(color: hintColor, fontSize: 11.5)),
              ],
            ),
          ],
          if (evidence.tags.isNotEmpty) ...[
            const SizedBox(height: 8),
            Wrap(
              spacing: 4,
              runSpacing: 4,
              children: [
                for (final tag in evidence.tags.take(6))
                  _MiniChip(label: tag, color: accent, isDark: isDark),
              ],
            ),
          ],
          if (onOpen != null) ...[
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: onOpen,
                style: OutlinedButton.styleFrom(
                  foregroundColor: accent,
                  side: BorderSide(color: accent.withValues(alpha: 0.7)),
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                icon: const Icon(Icons.travel_explore_rounded, size: 16),
                label: Text('打开$slotLabel',
                    style: const TextStyle(fontSize: 12.5)),
              ),
            ),
          ],
        ],
      ),
    );
  }

  String _shortDate(String iso) {
    if (iso.isEmpty) return '';
    final t = iso.replaceAll('T', ' ');
    return t.length >= 10 ? t.substring(0, 10) : t;
  }
}

class _MiniChip extends StatelessWidget {
  final String label;
  final Color color;
  final bool isDark;

  const _MiniChip({
    required this.label,
    required this.color,
    required this.isDark,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: isDark ? 0.18 : 0.12),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        label,
        style: TextStyle(color: color, fontSize: 10.5),
      ),
    );
  }
}

class _TimeCompareDiffPanel extends StatelessWidget {
  final CompareDiff diff;
  final bool isDark;
  final Color textColor;
  final Color hintColor;

  const _TimeCompareDiffPanel({
    required this.diff,
    required this.isDark,
    required this.textColor,
    required this.hintColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: isDark
            ? const Color(0xFF14181F)
            : BDDesign.colorAshGray.withValues(alpha: 0.4),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: BDDesign.colorFadedOlive.withValues(
            alpha: isDark ? 0.35 : 0.25,
          ),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.difference_outlined,
                  size: 15, color: BDDesign.colorFadedOlive),
              const SizedBox(width: 6),
              Text(
                '差异分析',
                style: TextStyle(
                  color: textColor,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          if (diff.hasObjectChanges ||
              diff.commonObjects.isNotEmpty) ...[
            const SizedBox(height: 8),
            _DiffSection(
              title: '物体',
              added: diff.addedObjects,
              removed: diff.removedObjects,
              common: diff.commonObjects,
              isDark: isDark,
              hintColor: hintColor,
            ),
          ],
          if (diff.hasTagChanges || diff.commonTags.isNotEmpty) ...[
            const SizedBox(height: 8),
            _DiffSection(
              title: '标签',
              added: diff.addedTags,
              removed: diff.removedTags,
              common: diff.commonTags,
              isDark: isDark,
              hintColor: hintColor,
            ),
          ],
          if (diff.limitations.isNotEmpty) ...[
            const SizedBox(height: 10),
            for (final note in diff.limitations)
              Padding(
                padding: const EdgeInsets.only(top: 2),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.info_outline,
                        size: 12, color: hintColor),
                    const SizedBox(width: 4),
                    Expanded(
                      child: Text(
                        note,
                        style: TextStyle(color: hintColor, fontSize: 11.5),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ],
      ),
    );
  }
}

class _DiffSection extends StatelessWidget {
  final String title;
  final List<String> added;
  final List<String> removed;
  final List<String> common;
  final bool isDark;
  final Color hintColor;

  const _DiffSection({
    required this.title,
    required this.added,
    required this.removed,
    required this.common,
    required this.isDark,
    required this.hintColor,
  });

  static const Color _addedColor = Color(0xFF4F8C5A);
  static const Color _removedColor = BDDesign.colorDarkRed;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: TextStyle(
            color: hintColor,
            fontSize: 11.5,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 4),
        Wrap(
          spacing: 4,
          runSpacing: 4,
          children: [
            for (final t in added)
              _DiffChip(label: '+ $t', color: _addedColor, isDark: isDark),
            for (final t in removed)
              _DiffChip(label: '- $t', color: _removedColor, isDark: isDark),
            for (final t in common.take(6))
              _DiffChip(
                  label: t,
                  color: BDDesign.colorMutedBlue,
                  isDark: isDark,
                  faded: true),
          ],
        ),
      ],
    );
  }
}

class _DiffChip extends StatelessWidget {
  final String label;
  final Color color;
  final bool isDark;
  final bool faded;

  const _DiffChip({
    required this.label,
    required this.color,
    required this.isDark,
    this.faded = false,
  });

  @override
  Widget build(BuildContext context) {
    final bgAlpha = faded ? (isDark ? 0.15 : 0.10) : (isDark ? 0.25 : 0.18);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: bgAlpha),
        borderRadius: BorderRadius.circular(5),
        border: faded
            ? null
            : Border.all(color: color.withValues(alpha: 0.45), width: 0.6),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: color,
          fontSize: 11,
          fontWeight: faded ? FontWeight.w400 : FontWeight.w500,
        ),
      ),
    );
  }
}
