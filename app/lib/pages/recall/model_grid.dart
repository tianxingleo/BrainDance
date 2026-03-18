import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../../configs/app_config.dart';
import '../../configs/motion_tokens.dart';

class RecallModelGrid extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final List<Map<String, dynamic>> models;
  final Map<String, dynamic>? activeModelAction;
  final GlobalKey Function(Map<String, dynamic>) modelCardKeyFor;
  final bool Function(Map<String, dynamic>?, Map<String, dynamic>?) isSameModel;
  final void Function(Map<String, dynamic>, dynamic) onNavigateToViewer;
  final void Function(Map<String, dynamic>) onShowModelActions;

  const RecallModelGrid({
    super.key,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkInput,
    required this.models,
    required this.activeModelAction,
    required this.modelCardKeyFor,
    required this.isSameModel,
    required this.onNavigateToViewer,
    required this.onShowModelActions,
  });

  @override
  Widget build(BuildContext context) {
    final textColor = isDark
        ? const Color(0xFFFFFFFF)
        : const Color(0xFF333333);
    final hintTextColor = isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;

    final isSearchWithFrames =
        models.isNotEmpty && models.first.containsKey('matched_frames');

    if (isSearchWithFrames) {
      return ListView.builder(
        padding: const EdgeInsets.fromLTRB(16.0, 6.0, 16.0, 16.0),
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        itemCount: models.length,
        itemBuilder: (context, index) {
          final model = models[index];
          final sceneId = model['scene_id'] ?? 'Unknown Scene';
          final desc = model['description'] ?? '没有描述信息';
          final similarity = model['similarity'] as double?;
          final userId = model['user_id'] ?? '';
          final matchedFrames = model['matched_frames'] as List<dynamic>? ?? [];

          return TweenAnimationBuilder<double>(
            tween: Tween(begin: 0.0, end: 1.0),
            duration:
                BDMotion.durationNormal +
                Duration(milliseconds: (index * 50).clamp(0, 400)),
            curve: BDMotion.curveEnter,
            builder: (context, value, child) {
              return Transform.translate(
                offset: Offset(0, 20 * (1 - value)),
                child: Opacity(opacity: value, child: child),
              );
            },
            child: Container(
              margin: const EdgeInsets.only(bottom: 16.0),
              decoration: BoxDecoration(
                color: isDark ? darkCard : BDDesign.colorPaperWhite,
                borderRadius: BDDesign.radiusLarge,
                boxShadow: isDark ? [] : [BDDesign.shadowLight],
                border: Border.all(
                  color: isDark ? const Color(0xFF2A2A30) : Colors.transparent,
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  GestureDetector(
                    onTap: () => onNavigateToViewer(model, null),
                    onLongPress: () => onShowModelActions(model),
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                TDText(
                                  sceneId,
                                  font: theme.fontTitleMedium,
                                  fontWeight: FontWeight.w600,
                                  maxLines: 1,
                                  textColor: textColor,
                                ),
                                const SizedBox(height: 4),
                                TDText(
                                  desc,
                                  font: theme.fontBodySmall,
                                  textColor: hintTextColor,
                                  maxLines: 2,
                                ),
                              ],
                            ),
                          ),
                          if (similarity != null)
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 4,
                              ),
                              decoration: BoxDecoration(
                                color: theme.brandColor4.withAlpha(220),
                                borderRadius: BorderRadius.circular(6),
                              ),
                              child: TDText(
                                '${(similarity * 100).toStringAsFixed(1)}%',
                                font: theme.fontBodySmall,
                                textColor: isDark
                                    ? const Color(0xFFFFFFFF)
                                    : Colors.white,
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                  if (matchedFrames.isNotEmpty)
                    SizedBox(
                      height: 120,
                      child: ListView.builder(
                        scrollDirection: Axis.horizontal,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16.0,
                        ).copyWith(bottom: 16.0),
                        itemCount: matchedFrames.length,
                        itemBuilder: (context, frameIndex) {
                          final frame = matchedFrames[frameIndex];
                          final imageName = frame['image_name'];
                          final transformMatrix = frame['transform_matrix'];
                          final frameSim = frame['similarity'] as double?;

                          final imageUrl = Supabase.instance.client.storage
                              .from('braindance-assets')
                              .getPublicUrl(
                                '$userId/$sceneId/output/images/$imageName',
                              );

                          return GestureDetector(
                            onTap: () =>
                                onNavigateToViewer(model, transformMatrix),
                            child: Container(
                              width: 140,
                              margin: const EdgeInsets.only(right: 12.0),
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(8.0),
                                color: isDark ? darkInput : theme.grayColor3,
                              ),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(8.0),
                                child: Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    Image.network(
                                      imageUrl,
                                      fit: BoxFit.cover,
                                      loadingBuilder:
                                          (context, child, loadingProgress) {
                                            if (loadingProgress == null) {
                                              return child;
                                            }
                                            return Center(
                                              child: CircularProgressIndicator(
                                                value:
                                                    loadingProgress.expectedTotalBytes !=
                                                        null
                                                    ? loadingProgress.cumulativeBytesLoaded /
                                                          loadingProgress.expectedTotalBytes!
                                                    : null,
                                              ),
                                            );
                                          },
                                      errorBuilder:
                                          (context, error, stackTrace) {
                                            return const Center(
                                              child: Icon(
                                                Icons.broken_image,
                                                color: Colors.grey,
                                              ),
                                            );
                                          },
                                    ),
                                    if (frameSim != null)
                                      Positioned(
                                        bottom: 4,
                                        left: 4,
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(
                                            horizontal: 4,
                                            vertical: 2,
                                          ),
                                          decoration: BoxDecoration(
                                            color: Colors.black.withAlpha(100),
                                            borderRadius: BorderRadius.circular(
                                              4,
                                            ),
                                          ),
                                          child: Text(
                                            '${(frameSim * 100).toStringAsFixed(1)}%',
                                            style: const TextStyle(
                                              color: Colors.white,
                                              fontSize: 10,
                                            ),
                                          ),
                                        ),
                                      ),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                ],
              ),
            ),
          );
        },
      );
    }

    return GridView.builder(
      padding: const EdgeInsets.only(
        left: 16.0,
        right: 16.0,
        top: 14.0,
        bottom: 16.0,
      ),
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 16.0,
        mainAxisSpacing: 16.0,
        childAspectRatio: 0.85,
      ),
      itemCount: models.length,
      itemBuilder: (context, index) {
        final model = models[index];
        final cardKey = modelCardKeyFor(model);
        final isActionTarget = isSameModel(activeModelAction, model);

        return TweenAnimationBuilder<double>(
          tween: Tween(begin: 0.0, end: 1.0),
          duration:
              BDMotion.durationNormal +
              Duration(milliseconds: (index * 50).clamp(0, 400)),
          curve: BDMotion.curveEnter,
          builder: (context, value, child) {
            return Transform.translate(
              offset: Offset(0, 20 * (1 - value)),
              child: Opacity(
                opacity: isActionTarget ? 0.0 : value,
                child: child,
              ),
            );
          },
          child: IgnorePointer(
            ignoring: isActionTarget,
            child: GestureDetector(
              onTap: () => onNavigateToViewer(model, null),
              onLongPressStart: (_) => onShowModelActions(model),
              child: Container(
                key: cardKey,
                child: RecallModelTile(
                  model: model,
                  theme: theme,
                  isDark: isDark,
                  darkCard: darkCard,
                  darkInput: darkInput,
                  textColor: textColor,
                  hintTextColor: hintTextColor,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

class RecallModelActionOverlay extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Map<String, dynamic> model;
  final Rect rect;
  final VoidCallback onDismiss;
  final void Function(Map<String, dynamic>, dynamic) onNavigateToViewer;
  final Future<void> Function(Map<String, dynamic>) onShareModelToCommunity;

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
    required this.onShareModelToCommunity,
  });

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    const horizontalGap = 12.0;
    const actionWidth = 112.0;
    final actionLeft = (rect.right + horizontalGap + actionWidth <=
            screenWidth - 16)
        ? rect.right + horizontalGap
        : rect.left + rect.width - actionWidth;

    return Positioned.fill(
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: onDismiss,
        child: Stack(
          children: [
            TweenAnimationBuilder<double>(
              tween: Tween(begin: 0, end: 1),
              duration: const Duration(milliseconds: 220),
              curve: Curves.easeOutCubic,
              builder: (context, value, child) {
                return Opacity(
                  opacity: value,
                  child: ClipRect(
                    child: BackdropFilter(
                      filter: ImageFilter.blur(
                        sigmaX: 10 * value,
                        sigmaY: 10 * value,
                      ),
                      child: Container(
                        color: Colors.black.withValues(alpha: 0.12 * value),
                      ),
                    ),
                  ),
                );
              },
            ),
            Positioned(
              left: rect.left,
              top: rect.top,
              width: rect.width,
              height: rect.height,
              child: TweenAnimationBuilder<double>(
                tween: Tween(begin: 0, end: 1),
                duration: const Duration(milliseconds: 240),
                curve: Curves.easeOutCubic,
                builder: (context, value, child) {
                  return Transform.translate(
                    offset: Offset(0, -10 * value),
                    child: Transform.scale(
                      scale: 1 + (0.045 * value),
                      alignment: Alignment.center,
                      child: child,
                    ),
                  );
                },
                child: GestureDetector(
                  onTap: () => onNavigateToViewer(model, null),
                  onLongPress: onDismiss,
                  child: RecallModelTile(
                    model: model,
                    theme: theme,
                    isDark: isDark,
                    darkCard: darkCard,
                    darkInput: darkInput,
                    textColor: isDark
                        ? const Color(0xFFFFFFFF)
                        : BDDesign.colorInkBlack,
                    hintTextColor: isDark
                        ? const Color(0xFF888888)
                        : theme.fontGyColor3,
                    elevated: true,
                  ),
                ),
              ),
            ),
            Positioned(
              left: actionLeft,
              top: rect.top + 24,
              child: TweenAnimationBuilder<double>(
                tween: Tween(begin: 0, end: 1),
                duration: const Duration(milliseconds: 260),
                curve: Curves.easeOutBack,
                builder: (context, value, child) {
                  return Opacity(
                    opacity: value,
                    child: Transform.translate(
                      offset: Offset(18 * (1 - value), 0),
                      child: child,
                    ),
                  );
                },
                child: Material(
                  color: Colors.transparent,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(18),
                    onTap: () async {
                      onDismiss();
                      await onShareModelToCommunity(model);
                    },
                    child: Ink(
                      width: actionWidth,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 12,
                      ),
                      decoration: BoxDecoration(
                        color: isDark
                            ? const Color(0xEE1F2430)
                            : Colors.white.withAlpha(236),
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(
                          color: isDark
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
                          Icon(
                            Icons.public_rounded,
                            size: 18,
                            color: isDark
                                ? BDDesign.colorPaperWhite
                                : BDDesign.colorInkBlack,
                          ),
                          const SizedBox(height: 6),
                          Text(
                            '分享到社区',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: 12.5,
                              height: 1.2,
                              fontWeight: FontWeight.w700,
                              color: isDark
                                  ? BDDesign.colorPaperWhite
                                  : BDDesign.colorInkBlack,
                            ),
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

class RecallModelTile extends StatelessWidget {
  final Map<String, dynamic> model;
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Color textColor;
  final Color hintTextColor;
  final bool elevated;

  const RecallModelTile({
    super.key,
    required this.model,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkInput,
    required this.textColor,
    required this.hintTextColor,
    this.elevated = false,
  });

  @override
  Widget build(BuildContext context) {
    final sceneId = model['scene_id'] ?? 'Unknown Scene';
    final desc = model['description'] ?? textLocalize("recall_no_desc");
    final similarity = model['similarity'] as double?;

    return Container(
      decoration: BoxDecoration(
        color: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
        borderRadius: BorderRadius.circular(28.0),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(elevated ? 46 : 20),
            blurRadius: elevated ? 26 : 10,
            spreadRadius: elevated ? 2 : 0,
            offset: Offset(0, elevated ? 16 : 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                Container(
                  decoration: BoxDecoration(
                    color: isDark ? darkInput : theme.grayColor3,
                    borderRadius: const BorderRadius.vertical(
                      top: Radius.circular(28.0),
                    ),
                  ),
                  clipBehavior: Clip.hardEdge,
                  child:
                      model['preview_img_path'] != null &&
                          model['preview_img_path'].toString().isNotEmpty
                      ? Image.network(
                          model['preview_img_path'],
                          fit: BoxFit.cover,
                          errorBuilder: (context, error, stackTrace) =>
                              RecallModelMockCover(
                                isDark: isDark,
                                theme: theme,
                              ),
                          loadingBuilder: (context, child, loadingProgress) {
                            if (loadingProgress == null) {
                              return child;
                            }
                            return const Center(
                              child: CircularProgressIndicator(),
                            );
                          },
                        )
                      : RecallModelMockCover(isDark: isDark, theme: theme),
                ),
                if (similarity != null)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 6,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: theme.brandColor4.withAlpha(220),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: TDText(
                        '${(similarity * 100).toStringAsFixed(1)}%',
                        font: theme.fontBodyExtraSmall,
                        textColor: isDark
                            ? const Color(0xFFFFFFFF)
                            : Colors.white,
                      ),
                    ),
                  ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(12.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TDText(
                  sceneId,
                  font: theme.fontTitleMedium,
                  fontWeight: FontWeight.w600,
                  maxLines: 1,
                  textColor: textColor,
                ),
                const SizedBox(height: 4),
                TDText(
                  desc,
                  font: theme.fontBodySmall,
                  textColor: hintTextColor,
                  maxLines: 2,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class RecallModelMockCover extends StatelessWidget {
  final bool isDark;
  final TDThemeData theme;

  const RecallModelMockCover({
    super.key,
    required this.isDark,
    required this.theme,
  });

  @override
  Widget build(BuildContext context) {
    final accent = isDark ? const Color(0xFF7AA2FF) : BDDesign.colorMutedBlue;

    return Container(
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF1A1E27) : const Color(0xFFF6F8FC),
        border: Border.all(
          color: isDark ? Colors.white.withAlpha(18) : accent.withAlpha(35),
        ),
      ),
      child: Center(
        child: Container(
          width: 60,
          height: 60,
          decoration: BoxDecoration(
            color: isDark
                ? Colors.white.withAlpha(6)
                : Colors.white.withAlpha(190),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: isDark ? Colors.white.withAlpha(18) : accent.withAlpha(28),
            ),
          ),
          child: Icon(
            Icons.auto_awesome_mosaic_rounded,
            size: 28,
            color: accent.withAlpha(210),
          ),
        ),
      ),
    );
  }
}
