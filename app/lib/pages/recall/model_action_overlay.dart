/// 模型操作浮层
///
/// 长按模型卡片时弹出的操作菜单浮层，包含查看详情、重命名、下载、删除、分享等操作。
library;

import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../../configs/app_config.dart';
import '../../configs/motion_tokens.dart';
import 'model_grid.dart';

class RecallModelActionOverlay extends StatefulWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Map<String, dynamic> model;
  final Rect rect;
  final VoidCallback onDismiss;
  final void Function(Map<String, dynamic>, dynamic) onNavigateToViewer;
  final Future<void> Function(Map<String, dynamic>) onShowModelDetails;
  final Future<void> Function(Map<String, dynamic>) onDownloadModel;
  final Future<void> Function(Map<String, dynamic>) onDeleteLocalModel;
  final Future<void> Function(Map<String, dynamic>) onShareModelToCommunity;
  final Future<void> Function(Map<String, dynamic>) onRenameModel;
  final Future<void> Function(Map<String, dynamic>) onDeleteCloudModel;
  final bool isLocalCached;
  final bool isOwnModel;

  const RecallModelActionOverlay({
    super.key,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkInput,
    required this.model,
    required this.rect,
    required this.onDismiss,
    required this.onNavigateToViewer,
    required this.onShowModelDetails,
    required this.onDownloadModel,
    required this.onDeleteLocalModel,
    required this.onShareModelToCommunity,
    required this.onRenameModel,
    required this.onDeleteCloudModel,
    required this.isLocalCached,
    required this.isOwnModel,
  });

  @override
  State<RecallModelActionOverlay> createState() =>
      RecallModelActionOverlayState();
}

class RecallModelActionOverlayState extends State<RecallModelActionOverlay>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _blurOpacityAnimation;
  late final Animation<double> _cardScaleAnimation;
  late final Animation<double> _cardTranslateAnimation;
  late final Animation<double> _menuOpacityAnimation;
  late final Animation<double> _menuTranslateAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 320),
      reverseDuration: const Duration(milliseconds: 240),
    );

    _blurOpacityAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
      reverseCurve: Curves.easeInCubic,
    );

    _cardScaleAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutBack,
      reverseCurve: Curves.easeInCubic,
    );

    _cardTranslateAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
      reverseCurve: Curves.easeInCubic,
    );

    _menuOpacityAnimation = CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.1, 1.0, curve: Curves.easeOutCubic),
      reverseCurve: Curves.easeInCubic,
    );

    _menuTranslateAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutBack,
      reverseCurve: Curves.easeInCubic,
    );

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> hide() async {
    if (mounted) {
      await _controller.reverse();
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.sizeOf(context).width;
    const screenPadding = 16.0;
    const horizontalGap = 12.0;
    const actionWidth = 128.0;
    final maxLeft = screenWidth - screenPadding - actionWidth;
    final actionLeft = (widget.rect.right + horizontalGap)
        .clamp(screenPadding, maxLeft)
        .toDouble();

    return Positioned.fill(
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: widget.onDismiss,
        child: Stack(
          children: [
            AnimatedBuilder(
              animation: _blurOpacityAnimation,
              child: const _OverlayBlurScrim(),
              builder: (context, child) {
                return Opacity(
                  opacity: _blurOpacityAnimation.value,
                  child: child,
                );
              },
            ),
            Positioned(
              left: widget.rect.left,
              top: widget.rect.top,
              width: widget.rect.width,
              height: widget.rect.height,
              child: AnimatedBuilder(
                animation: _controller,
                builder: (context, child) {
                  final tValue = _cardTranslateAnimation.value;
                  final sValue = _cardScaleAnimation.value;
                  return Transform.translate(
                    offset: Offset(0, -10 * tValue),
                    child: Transform.scale(
                      scale: 1 + (0.045 * sValue),
                      alignment: Alignment.center,
                      child: GestureDetector(
                        onTap: () =>
                            widget.onNavigateToViewer(widget.model, null),
                        onLongPress: widget.onDismiss,
                        child: RepaintBoundary(
                          child: RecallModelTile(
                            model: widget.model,
                            theme: widget.theme,
                            isDark: widget.isDark,
                            darkCard: widget.darkCard,
                            darkInput: widget.darkInput,
                            textColor: widget.isDark
                                ? const Color(0xFFFFFFFF)
                                : BDDesign.colorInkBlack,
                            hintTextColor: widget.isDark
                                ? const Color(0xFF888888)
                                : widget.theme.fontGyColor3,
                            elevated: true,
                            elevationProgress: sValue,
                            imageOnly: widget.model['_imageOnly'] == true,
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
            Positioned.fill(
              child: AnimatedBuilder(
                animation: _blurOpacityAnimation,
                builder: (context, child) {
                  final bValue = _blurOpacityAnimation.value;
                  return IgnorePointer(
                    child: Opacity(
                      opacity: bValue,
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              widget.theme.brandColor4.withValues(
                                alpha: widget.isDark ? 0.25 : 0.15,
                              ),
                              Colors.transparent,
                            ],
                            begin: Alignment.centerLeft,
                            end: Alignment.centerRight,
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
            Positioned(
              left: actionLeft,
              top: widget.rect.top + 24,
              child: AnimatedBuilder(
                animation: _controller,
                builder: (context, child) {
                  final mValue = _menuTranslateAnimation.value;
                  final oValue = _menuOpacityAnimation.value;
                  return Opacity(
                    opacity: oValue.clamp(0.0, 1.0),
                    child: Transform.translate(
                      offset: Offset(18 * (1 - mValue), 0),
                      child: child,
                    ),
                  );
                },
                child: Material(
                  color: Colors.transparent,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(18),
                    onTap: () {
                      widget.onDismiss();
                    },
                    child: Ink(
                      width: actionWidth,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 12,
                      ),
                      decoration: BoxDecoration(
                        color: widget.isDark
                            ? const Color(0xEE1F2430)
                            : Colors.white.withAlpha(236),
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(
                          color: widget.isDark
                              ? Colors.white.withValues(alpha: 0.08)
                              : BDDesign.colorMutedBlue.withValues(alpha: 0.14),
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withAlpha(20),
                            blurRadius: 18,
                            offset: const Offset(0, 10),
                          ),
                        ],
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          ActionMenuItem(
                            icon: Icons.info_outline_rounded,
                            label: textLocalize("recall_info"),
                            isDark: widget.isDark,
                            onTap: () async {
                              widget.onDismiss();
                              await widget.onShowModelDetails(widget.model);
                            },
                          ),
                          const SizedBox(height: 6),
                          if (widget.isOwnModel)
                            ActionMenuItem(
                              icon: Icons.edit_rounded,
                              label: textLocalize("recall_rename"),
                              isDark: widget.isDark,
                              onTap: () async {
                                widget.onDismiss();
                                await widget.onRenameModel(widget.model);
                              },
                            ),
                          if (widget.isOwnModel) const SizedBox(height: 6),
                          if (widget.isLocalCached)
                            ActionMenuItem(
                              icon: Icons.delete_outline_rounded,
                              label: textLocalize('recall_delete_local'),
                              isDark: widget.isDark,
                              destructive: true,
                              onTap: () async {
                                widget.onDismiss();
                                final confirmed =
                                    await _showDeleteConfirmDialog(context);
                                if (confirmed == true) {
                                  await widget.onDeleteLocalModel(widget.model);
                                }
                              },
                            )
                          else
                            ActionMenuItem(
                              icon: Icons.download_rounded,
                              label: textLocalize('recall_download_model'),
                              isDark: widget.isDark,
                              onTap: () async {
                                widget.onDismiss();
                                widget.onNavigateToViewer(widget.model, null);
                              },
                            ),
                          if (widget.isOwnModel) ...[
                            const SizedBox(height: 6),
                            ActionMenuItem(
                              icon: Icons.delete_outline_rounded,
                              label: textLocalize('recall_delete_cloud'),
                              isDark: widget.isDark,
                              destructive: true,
                              onTap: () async {
                                widget.onDismiss();
                                final confirmed =
                                    await _showDeleteConfirmDialog(context);
                                if (confirmed == true) {
                                  await widget.onDeleteCloudModel(widget.model);
                                }
                              },
                            ),
                          ],
                          const SizedBox(height: 6),
                          ActionMenuItem(
                            icon: Icons.public_rounded,
                            label: textLocalize('recall_share_community'),
                            isDark: widget.isDark,
                            onTap: () async {
                              widget.onDismiss();
                              await widget.onShareModelToCommunity(
                                widget.model,
                              );
                            },
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _OverlayBlurScrim extends StatelessWidget {
  const _OverlayBlurScrim();

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: ClipRect(
        child: BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
          child: ColoredBox(color: Color(0x1F000000)),
        ),
      ),
    );
  }
}

/// 操作菜单单项
class ActionMenuItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool isDark;
  final bool destructive;
  final Future<void> Function() onTap;

  const ActionMenuItem({
    super.key,
    required this.icon,
    required this.label,
    required this.isDark,
    required this.onTap,
    this.destructive = false,
  });

  @override
  Widget build(BuildContext context) {
    final color = destructive
        ? const Color(0xFFD34C4C)
        : (isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack);

    return InkWell(
      borderRadius: BorderRadius.circular(12),
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 2, vertical: 8),
        child: Row(
          children: [
            Icon(icon, size: 18, color: color),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                label,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: color,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

Future<bool?> _showDeleteConfirmDialog(BuildContext context) {
  final isDark = Theme.of(context).brightness == Brightness.dark;
  return showDialog<bool>(
    context: context,
    barrierDismissible: false,
    builder: (ctx) {
      return AlertDialog(
        backgroundColor: isDark ? const Color(0xFF1C1C1E) : Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Text(
          textLocalize('recall_delete_confirm_title'),
          style: TextStyle(
            color: isDark ? Colors.white : Colors.black87,
            fontWeight: FontWeight.w600,
          ),
        ),
        content: Text(
          textLocalize('recall_delete_confirm_message'),
          style: TextStyle(
            color: isDark ? Colors.white70 : Colors.black54,
            height: 1.4,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text(
              textLocalize('recall_delete_confirm_cancel'),
              style: TextStyle(color: isDark ? Colors.white70 : Colors.black54),
            ),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            style: TextButton.styleFrom(foregroundColor: Colors.redAccent),
            child: Text(textLocalize('recall_delete_confirm_yes')),
          ),
        ],
      );
    },
  );
}
