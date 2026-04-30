/// 模型网格主文件
///
/// 包含 [RecallModelGrid]（主网格视图）、[RecallModelTile]（模型卡片）和
/// [RecallModelMockCover]（占位封面）三个公开组件。
///
/// 其他子模块拆分至：
/// - [model_grid_helpers.dart]  工具函数与常量
/// - [model_action_overlay.dart] 操作浮层
/// - [adaptive_thumbnail.dart]  自适应缩略图
/// - [time_peeling.dart]        时间线剥离

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../../configs/app_config.dart';
import '../../configs/motion_tokens.dart';
import 'adaptive_thumbnail.dart';
import 'model_card.dart';
import 'model_grid_helpers.dart';

/// 主模型网格
///
/// 根据搜索结果是否包含 [matched_frames] 字段自动切换列表/网格布局。
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
  final void Function(Map<String, dynamic> model, {bool imageOnly}) onShowModelActions;
  final String Function(String) toPublicUrl;

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
    required this.toPublicUrl,
  });

  @override
  Widget build(BuildContext context) {
    final textColor = resolveTextColor(isDark);
    final hintTextColor = resolveHintTextColor(isDark, theme);

    final isSearchWithFrames =
        models.isNotEmpty && models.first.containsKey('matched_frames');

    if (isSearchWithFrames) {
      return SliverPadding(
        padding: const EdgeInsets.fromLTRB(16.0, 6.0, 16.0, 16.0),
        sliver: SliverList(
          delegate: SliverChildBuilderDelegate(
            (context, index) {
              final model = models[index];
              final cardKey = modelCardKeyFor(model);
              final isActionTarget = isSameModel(activeModelAction, model);
              final displayName = modelDisplayName(model);
              final sceneStorageId = model['scene_id']?.toString() ?? '';
              final desc =
                  model['description'] ?? textLocalize("recall_no_desc");
              final similarity = model['similarity'] as double?;
              final userId = model['user_id'] ?? '';
              final matchedFrames =
                  model['matched_frames'] as List<dynamic>? ?? [];

              return RepaintBoundary(
                child: IgnorePointer(
                  ignoring: isActionTarget,
                  child: Opacity(
                    opacity: isActionTarget ? 0.0 : 1.0,
                    child: Container(
                      key: cardKey,
                      margin: const EdgeInsets.only(bottom: 16.0),
                      decoration: BoxDecoration(
                        color: isDark ? darkCard : BDDesign.colorPaperWhite,
                        borderRadius: BDDesign.radiusLarge,
                        boxShadow: isDark ? [] : [BDDesign.shadowLight],
                        border: Border.all(
                          color: isDark
                              ? const Color(0xFF2A2A30)
                              : Colors.transparent,
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          GestureDetector(
                            onTap: () => onNavigateToViewer(model, null),
                            onLongPressStart: (_) => onShowModelActions(model, imageOnly: false), // Here we are sending false for standard view cards
                            child: Padding(
                              padding: const EdgeInsets.all(16.0),
                              child: Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        TDText(
                                          displayName,
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
                                    buildSimilarityBadge(
                                      child: Container(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 8,
                                          vertical: 4,
                                        ),
                                        decoration: BoxDecoration(
                                          color: theme.brandColor4.withAlpha(
                                            220,
                                          ),
                                          borderRadius: BorderRadius.circular(
                                            6,
                                          ),
                                        ),
                                        child: TDText(
                                          formatSimilarity(similarity),
                                          font: theme.fontBodySmall,
                                          textColor: isDark
                                              ? const Color(0xFFFFFFFF)
                                              : Colors.white,
                                        ),
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
                                  final frameSim =
                                      frame['similarity'] as double?;

                                  final imageUrl = Supabase
                                      .instance
                                      .client
                                      .storage
                                      .from('braindance-assets')
                                      .getPublicUrl(
                                        '$userId/$sceneStorageId/output/images/$imageName',
                                      );

                                  return GestureDetector(
                                    onTap: () =>
                                        onNavigateToViewer(model, frame),
                                    child: AdaptiveFrameThumbnail(
                                      imageUrl: imageUrl,
                                      frameSim: frameSim,
                                      height: 104,
                                      backgroundColor: isDark
                                          ? darkInput
                                          : theme.grayColor3,
                                    ),
                                  );
                                },
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                ),
              );
            },
            childCount: models.length,
            addAutomaticKeepAlives: false,
          ),
        ),
      );
    }

    return SliverPadding(
      padding: const EdgeInsets.only(
        left: 16.0,
        right: 16.0,
        top: 14.0,
        bottom: 16.0,
      ),
      sliver: SliverGrid(
        delegate: SliverChildBuilderDelegate(
          (context, index) {
            final model = models[index];
            final cardKey = modelCardKeyFor(model);
            final isActionTarget = isSameModel(activeModelAction, model);

            return RepaintBoundary(
              child: IgnorePointer(
                ignoring: isActionTarget,
                child: Opacity(
                  opacity: isActionTarget ? 0.0 : 1.0,
                  child: GestureDetector(
                    onTap: () => onNavigateToViewer(model, null),
                    onLongPressStart: (_) => onShowModelActions(model, imageOnly: false), // Here we are sending false for standard view cards
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
                        toPublicUrl: toPublicUrl,
                      ),
                    ),
                  ),
                ),
              ),
            );
          },
          childCount: models.length,
          addAutomaticKeepAlives: false,
        ),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          crossAxisSpacing: 16.0,
          mainAxisSpacing: 16.0,
          childAspectRatio: 0.85,
        ),
      ),
    );
  }
}

/// 模型卡片瓦片
///
/// 展示单个模型的封面图、名称、描述和下载状态。
class RecallModelTile extends StatelessWidget {
  final Map<String, dynamic> model;
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Color textColor;
  final Color hintTextColor;
  final bool elevated;
  final double? elevationProgress;
  final String Function(String)? toPublicUrl;
  final bool imageOnly;

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
    this.elevationProgress,
    this.toPublicUrl,
    this.imageOnly = false,
  });

  @override
  Widget build(BuildContext context) {
    final sceneId = modelDisplayName(model);
    final desc = model['description'] ?? textLocalize("recall_no_desc");
    final similarity = model['similarity'] as double?;
    final plyPath = model['ply_path'] as String? ?? '';
    final modelUrl = plyPath.isNotEmpty && toPublicUrl != null
        ? toPublicUrl!(plyPath)
        : './models/scene_auto_sync_raw.ply';
    final radius = BorderRadius.circular(28.0);

    return Container(
      decoration: BoxDecoration(
        color: isDark ? darkCard : theme.whiteColor1.withAlpha(220),
        borderRadius: radius,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(
              elevationProgress != null
                  ? (elevationProgress! > 0 ? (20 + (46 - 20) * elevationProgress!).round() : 20)
                  : (elevated ? 46 : 20),
            ),
            blurRadius: elevationProgress != null
                ? (10 + (26 - 10) * elevationProgress!)
                : (elevated ? 26 : 10),
            spreadRadius: elevationProgress != null
                ? (0 + (2 - 0) * elevationProgress!)
                : (elevated ? 2 : 0),
            offset: Offset(
              0,
              elevationProgress != null
                  ? (4 + (16 - 4) * elevationProgress!)
                  : (elevated ? 16 : 4),
            ),
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
                    borderRadius: imageOnly
                        ? radius
                        : const BorderRadius.vertical(
                            top: Radius.circular(28.0),
                          ),
                  ),
                  clipBehavior: Clip.hardEdge,
                  child:
                      model['preview_img_path'] != null &&
                          model['preview_img_path'].toString().isNotEmpty
                      ? CoverNetworkImage(
                          imageUrl: model['preview_img_path'].toString(),
                          backgroundColor: isDark
                              ? darkInput
                              : theme.grayColor3,
                          errorWidget: RecallModelMockCover(
                            isDark: isDark,
                            theme: theme,
                          ),
                        )
                      : RecallModelMockCover(isDark: isDark, theme: theme),
                ),
                if (similarity != null)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: buildSimilarityBadge(
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
                          formatSimilarity(similarity),
                          font: theme.fontBodyExtraSmall,
                          textColor: isDark
                              ? const Color(0xFFFFFFFF)
                              : Colors.white,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          if (!imageOnly)
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
                  if (toPublicUrl != null) ...[
                    const SizedBox(height: 6),
                    ModelDownloadBadge(
                      modelUrl: modelUrl,
                      isDark: isDark,
                      theme: theme,
                    ),
                  ],
                ],
              ),
            ),
        ],
      ),
    );
  }
}

/// 模型占位封面
///
/// 当模型无预览图时显示的占位组件。
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
