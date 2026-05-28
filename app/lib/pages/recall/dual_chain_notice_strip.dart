import 'package:flutter/material.dart';

import '../../configs/app_config.dart';
import '../../widgets/bd_surfaces.dart';

/// 一条慢链完成的提示。
class DualChainNotice {
  final String sceneId;
  final String displayName;
  final DateTime arrivedAt;

  const DualChainNotice({
    required this.sceneId,
    required this.displayName,
    required this.arrivedAt,
  });
}

/// Recall 页慢链完成时插入的通知行列表。
/// 单条样式：图标 + 标题（高亮 displayName）+ 刷新按钮 + 关闭按钮。
class DualChainNoticeStrip extends StatelessWidget {
  final List<DualChainNotice> notices;
  final bool isDark;

  /// 用户点击「刷新」时调用。需要消费方负责重新拉取模型并 dismiss。
  final ValueChanged<DualChainNotice> onRefresh;

  /// 用户点击「×」时调用。仅 dismiss，不刷新。
  final ValueChanged<DualChainNotice> onDismiss;

  const DualChainNoticeStrip({
    super.key,
    required this.notices,
    required this.isDark,
    required this.onRefresh,
    required this.onDismiss,
  });

  @override
  Widget build(BuildContext context) {
    if (notices.isEmpty) return const SizedBox.shrink();

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
      child: Column(
        children: [
          for (final notice in notices)
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: _DualChainNoticeRow(
                notice: notice,
                isDark: isDark,
                onRefresh: () => onRefresh(notice),
                onDismiss: () => onDismiss(notice),
              ),
            ),
        ],
      ),
    );
  }
}

class _DualChainNoticeRow extends StatelessWidget {
  final DualChainNotice notice;
  final bool isDark;
  final VoidCallback onRefresh;
  final VoidCallback onDismiss;

  const _DualChainNoticeRow({
    required this.notice,
    required this.isDark,
    required this.onRefresh,
    required this.onDismiss,
  });

  @override
  Widget build(BuildContext context) {
    final accent = const Color(0xFFFFA726); // 与现有 quality score 中档色一致
    final textColor = isDark ? Colors.white.withValues(alpha: 0.92) : const Color(0xFF1F2A36);
    final hintColor = isDark ? Colors.white.withValues(alpha: 0.6) : const Color(0xFF6E7888);

    final raw = textLocalize('dual_chain_strip_title');
    final title = raw.replaceAll('[NAME]', notice.displayName);

    return BDGlassSurface(
      noBlur: true,
      variant: BDGlassVariant.panel,
      borderRadius: const BorderRadius.all(Radius.circular(16)),
      tintColor: accent.withValues(alpha: isDark ? 0.14 : 0.10),
      borderColor: accent.withValues(alpha: isDark ? 0.32 : 0.28),
      padding: const EdgeInsets.fromLTRB(12, 8, 6, 8),
      child: Row(
        children: [
          Container(
            width: 28,
            height: 28,
            decoration: BoxDecoration(
              color: accent.withValues(alpha: isDark ? 0.22 : 0.16),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(Icons.auto_awesome_rounded, size: 16, color: accent),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              title,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: textColor,
                height: 1.3,
              ),
            ),
          ),
          TextButton(
            style: TextButton.styleFrom(
              minimumSize: const Size(48, 32),
              padding: const EdgeInsets.symmetric(horizontal: 10),
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              foregroundColor: accent,
            ),
            onPressed: onRefresh,
            child: Text(
              textLocalize('dual_chain_strip_action_refresh'),
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
            ),
          ),
          IconButton(
            visualDensity: VisualDensity.compact,
            iconSize: 18,
            padding: const EdgeInsets.all(4),
            constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
            onPressed: onDismiss,
            icon: Icon(Icons.close_rounded, color: hintColor),
            tooltip: 'Dismiss',
          ),
        ],
      ),
    );
  }
}
