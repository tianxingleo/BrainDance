import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../../configs/motion_tokens.dart';
import '../../services/preview_image_resolver.dart';
import '../../widgets/animated_network_image.dart';

class AgentAssetCard extends StatelessWidget {
  final String? displayName;
  final String description;
  final List<String> tags;
  final String? previewImgPath;
  final String? previewWebpPath;
  final double? score;
  final String scoreLabel;
  final bool isDark;
  final VoidCallback? onOpen;
  final String actionLabel;

  const AgentAssetCard({
    super.key,
    this.displayName,
    required this.description,
    this.tags = const [],
    this.previewImgPath,
    this.previewWebpPath,
    this.score,
    this.scoreLabel = '置信度',
    required this.isDark,
    this.onOpen,
    this.actionLabel = '打开场景',
  });

  PreviewImagePaths get _thumbnailPaths {
    return resolvePreviewImagePaths({
      'preview_img_path': previewImgPath,
      'preview_webp_path': previewWebpPath,
    }, normalize: _normalizeAssetUrl);
  }

  String _normalizeAssetUrl(String raw) {
    raw = raw.trim();
    const marker = '/storage/v1/object/public/braindance-assets/';
    final idx = raw.indexOf(marker);
    if (idx >= 0) raw = raw.substring(idx + marker.length);
    if (raw.startsWith('http://') || raw.startsWith('https://')) return raw;
    try {
      if (raw.startsWith('/')) raw = raw.substring(1);
      return Supabase.instance.client.storage
          .from('braindance-assets')
          .getPublicUrl(raw);
    } catch (_) {
      return raw;
    }
  }

  void _showDetailDialog(BuildContext context) {
    final paths = _thumbnailPaths;
    showGeneralDialog(
      context: context,
      barrierDismissible: true,
      barrierLabel: 'AssetDetail',
      transitionDuration: const Duration(milliseconds: 250),
      pageBuilder: (ctx, anim, secondAnim) {
        return _AssetDetailDialog(
          imageUrl: paths.primary,
          fallbackImageUrl: paths.fallback,
          displayName: displayName,
          description: description,
          tags: tags,
          score: score,
          scoreLabel: scoreLabel,
          isDark: isDark,
          onOpen: onOpen,
          actionLabel: actionLabel,
        );
      },
      transitionBuilder: (ctx, anim, secondAnim, child) {
        return BackdropFilter(
          filter: ImageFilter.blur(
            sigmaX: 5 * anim.value,
            sigmaY: 5 * anim.value,
          ),
          child: FadeTransition(opacity: anim, child: child),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final paths = _thumbnailPaths;
    final cardColor = isDark
        ? Colors.white.withValues(alpha: 0.06)
        : Colors.white.withValues(alpha: 0.85);
    final textColor = isDark ? Colors.white : const Color(0xFF1A1A2E);
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.6)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.8);

    return GestureDetector(
      onTap: () => _showDetailDialog(context),
      child: Container(
        decoration: BoxDecoration(
          color: cardColor,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isDark
                ? Colors.white.withValues(alpha: 0.08)
                : Colors.black.withValues(alpha: 0.06),
          ),
        ),
        clipBehavior: Clip.antiAlias,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SizedBox(
              height: 120,
              child: paths.primary != null
                  ? BDFadeInNetworkImage(
                      imageUrl: paths.primary!,
                      fallbackImageUrl: paths.fallback,
                      placeholder: _buildPlaceholder(loading: true),
                      errorWidget: _buildPlaceholder(),
                      fit: BoxFit.cover,
                    )
                  : _buildPlaceholder(),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 10, 12, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          displayName ?? description,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: textColor,
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      if (score != null)
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color: BDDesign.colorMutedBlue.withValues(
                              alpha: 0.15,
                            ),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            '$scoreLabel ${(score! * 100).toStringAsFixed(0)}%',
                            style: TextStyle(
                              color: BDDesign.colorMutedBlue,
                              fontSize: 11,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                    ],
                  ),
                  if (tags.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    _buildTags(hintColor),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPlaceholder({bool loading = false}) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: isDark
              ? [const Color(0xFF2A2A4A), const Color(0xFF1A1A2E)]
              : [const Color(0xFFE8EAF6), const Color(0xFFC5CAE9)],
        ),
      ),
      child: Center(
        child: loading
            ? SizedBox(
                width: 24,
                height: 24,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.3)
                      : Colors.black.withValues(alpha: 0.2),
                ),
              )
            : Icon(
                Icons.view_in_ar_rounded,
                size: 36,
                color: isDark
                    ? Colors.white.withValues(alpha: 0.2)
                    : Colors.black.withValues(alpha: 0.15),
              ),
      ),
    );
  }

  Widget _buildTags(Color hintColor) {
    final displayTags = tags.take(3).toList();
    final overflow = tags.length - 3;

    return Wrap(
      spacing: 6,
      runSpacing: 4,
      children: [
        for (final tag in displayTags)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.08)
                  : Colors.black.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(tag, style: TextStyle(color: hintColor, fontSize: 11)),
          ),
        if (overflow > 0)
          Text('+$overflow', style: TextStyle(color: hintColor, fontSize: 11)),
      ],
    );
  }
}

class _AssetDetailDialog extends StatelessWidget {
  final String? imageUrl;
  final String? fallbackImageUrl;
  final String? displayName;
  final String description;
  final List<String> tags;
  final double? score;
  final String scoreLabel;
  final bool isDark;
  final VoidCallback? onOpen;
  final String actionLabel;

  const _AssetDetailDialog({
    this.imageUrl,
    this.fallbackImageUrl,
    this.displayName,
    required this.description,
    required this.tags,
    this.score,
    this.scoreLabel = '置信度',
    required this.isDark,
    this.onOpen,
    this.actionLabel = '打开场景',
  });

  @override
  Widget build(BuildContext context) {
    final cardColor = isDark ? const Color(0xFF1E1E2E) : Colors.white;
    final textColor = isDark ? Colors.white : const Color(0xFF1A1A2E);
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.6)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.8);

    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 360),
        child: Material(
          color: Colors.transparent,
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 24),
            decoration: BoxDecoration(
              color: cardColor,
              borderRadius: BorderRadius.circular(16),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.3),
                  blurRadius: 24,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            clipBehavior: Clip.antiAlias,
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Stack(
                    children: [
                      SizedBox(
                        height: 180,
                        width: double.infinity,
                        child: imageUrl != null
                            ? BDFadeInNetworkImage(
                                imageUrl: imageUrl!,
                                fallbackImageUrl: fallbackImageUrl,
                                placeholder: _placeholder(),
                                errorWidget: _placeholder(),
                                fit: BoxFit.cover,
                              )
                            : _placeholder(),
                      ),
                      Positioned(
                        top: 8,
                        right: 8,
                        child: GestureDetector(
                          onTap: () => Navigator.of(context).pop(),
                          child: Container(
                            width: 32,
                            height: 32,
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: 0.5),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(
                              Icons.close_rounded,
                              color: Colors.white,
                              size: 18,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                  Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                displayName ?? '未命名资产',
                                style: TextStyle(
                                  color: textColor,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                            if (score != null)
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 8,
                                  vertical: 3,
                                ),
                                decoration: BoxDecoration(
                                  color: BDDesign.colorMutedBlue.withValues(
                                    alpha: 0.15,
                                  ),
                                  borderRadius: BorderRadius.circular(6),
                                ),
                                child: Text(
                                  '$scoreLabel ${(score! * 100).toStringAsFixed(0)}%',
                                  style: TextStyle(
                                    color: BDDesign.colorMutedBlue,
                                    fontSize: 12,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ),
                          ],
                        ),
                        if (description.isNotEmpty) ...[
                          const SizedBox(height: 10),
                          Text(
                            description,
                            style: TextStyle(
                              color: hintColor,
                              fontSize: 13,
                              height: 1.5,
                            ),
                          ),
                        ],
                        if (tags.isNotEmpty) ...[
                          const SizedBox(height: 12),
                          Wrap(
                            spacing: 6,
                            runSpacing: 6,
                            children: tags
                                .map(
                                  (tag) => Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 8,
                                      vertical: 4,
                                    ),
                                    decoration: BoxDecoration(
                                      color: isDark
                                          ? Colors.white.withValues(alpha: 0.08)
                                          : Colors.black.withValues(
                                              alpha: 0.05,
                                            ),
                                      borderRadius: BorderRadius.circular(6),
                                    ),
                                    child: Text(
                                      tag,
                                      style: TextStyle(
                                        color: hintColor,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ),
                                )
                                .toList(),
                          ),
                        ],
                        if (onOpen != null) ...[
                          const SizedBox(height: 16),
                          SizedBox(
                            width: double.infinity,
                            height: 40,
                            child: ElevatedButton.icon(
                              style: ElevatedButton.styleFrom(
                                backgroundColor: BDDesign.colorMutedBlue,
                                foregroundColor: Colors.white,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                elevation: 0,
                              ),
                              onPressed: () {
                                Navigator.of(context).pop();
                                onOpen!();
                              },
                              icon: const Icon(
                                Icons.open_in_new_rounded,
                                size: 16,
                              ),
                              label: Text(
                                actionLabel,
                                style: const TextStyle(fontSize: 14),
                              ),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _placeholder() {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: isDark
              ? [const Color(0xFF2A2A4A), const Color(0xFF1A1A2E)]
              : [const Color(0xFFE8EAF6), const Color(0xFFC5CAE9)],
        ),
      ),
      child: Center(
        child: Icon(
          Icons.view_in_ar_rounded,
          size: 48,
          color: isDark
              ? Colors.white.withValues(alpha: 0.2)
              : Colors.black.withValues(alpha: 0.15),
        ),
      ),
    );
  }
}
