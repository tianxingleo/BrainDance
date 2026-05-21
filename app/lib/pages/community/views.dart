import 'dart:math' as math;

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/pages/community/map_marker.dart';
import 'package:braindance/pages/community/map_page.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:braindance/widgets/animated_network_image.dart';
import 'package:flutter/material.dart';

import 'models.dart';

// ============================================================
// Explore Tab
// ============================================================

class CommunityExploreView extends StatelessWidget {
  final List<CommunityPost> posts;
  final int totalPosts;
  final int viewportPosts;
  final CommunityMapViewport mapViewport;
  final List<CommunityMapMarker> mapMarkers;
  final VoidCallback onOpenMap;
  final ValueChanged<CommunityPost> onTapPost;
  final List<String> availableTags;
  final String? selectedTag;
  final ValueChanged<String> onToggleTag;
  final VoidCallback onClearFilters;
  final double tagRadiusKm;

  const CommunityExploreView({
    super.key,
    required this.posts,
    required this.mapViewport,
    required this.onOpenMap,
    required this.onTapPost,
    this.totalPosts = 0,
    this.viewportPosts = 0,
    this.mapMarkers = const [],
    this.availableTags = const [],
    this.selectedTag,
    required this.onToggleTag,
    required this.onClearFilters,
    this.tagRadiusKm = 0,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    if (totalPosts == 0) return const _CommunityEmptyState();
    final boundsReady = mapViewport.bounds != null;
    final hasTag = selectedTag != null && selectedTag!.isNotEmpty;

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 104),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: CommunityMetricCard(
                  label: textLocalize('community_label_memories'),
                  value: '${posts.length}',
                  hint: textLocalize('community_label_memories_hint'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: CommunityMetricCard(
                  label: textLocalize('community_label_nodes'),
                  value:
                      '${posts.map((p) => p.placeName).toSet().length}',
                  hint: textLocalize('community_label_nodes_hint'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          BDPanelCard(
            padding: const EdgeInsets.fromLTRB(14, 14, 14, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'World Memory Map',
                            style: TextStyle(
                              color: textColor,
                              fontSize: 18,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            textLocalize('community_map_hint'),
                            style: TextStyle(color: hintColor, height: 1.35),
                          ),
                        ],
                      ),
                    ),
                    BDStatusPill(
                      label: 'ZOOM ${mapViewport.zoom}',
                      icon: Icons.public_rounded,
                      color: BDDesign.colorMutedBlue,
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                LayoutBuilder(
                  builder: (context, constraints) {
                    final mapWidth = constraints.maxWidth;
                    final mapHeight = math.max(240.0, mapWidth * 0.58);
                    return GestureDetector(
                      onTap: onOpenMap,
                      child: SizedBox(
                        height: mapHeight,
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(26),
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              CommunityAmapPreview(
                                viewport: mapViewport,
                                width: mapWidth.round().clamp(320, 1024),
                                height: mapHeight.round().clamp(240, 1024),
                                markers: mapMarkers,
                              ),
                              Positioned(
                                right: 12,
                                top: 12,
                                child: DecoratedBox(
                                  decoration: BoxDecoration(
                                    color: (isDark
                                            ? Colors.black
                                            : Colors.white)
                                        .withValues(alpha: 0.82),
                                    borderRadius: BorderRadius.circular(999),
                                    border: Border.all(
                                      color: isDark
                                          ? Colors.white.withValues(alpha: 0.08)
                                          : BDDesign.colorMutedBlue
                                              .withValues(alpha: 0.10),
                                    ),
                                  ),
                                  child: Padding(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 12,
                                      vertical: 8,
                                    ),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Icon(
                                          Icons.open_in_full_rounded,
                                          color: hintColor,
                                          size: 16,
                                        ),
                                        const SizedBox(width: 6),
                                        Text(
                                          '调整地图',
                                          style: TextStyle(
                                            color: hintColor,
                                            fontWeight: FontWeight.w700,
                                            fontSize: 12.5,
                                          ),
                                        ),
                                      ],
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
              ],
            ),
          ),
          const SizedBox(height: 14),
          _ExploreFilterBar(
            isDark: isDark,
            textColor: textColor,
            hintColor: hintColor,
            boundsReady: boundsReady,
            zoom: mapViewport.zoom,
            tagRadiusKm: tagRadiusKm,
            visibleCount: viewportPosts,
            totalCount: totalPosts,
            filteredCount: posts.length,
            availableTags: availableTags,
            selectedTag: selectedTag,
            onToggleTag: onToggleTag,
            onClearFilters: onClearFilters,
          ),
          const SizedBox(height: 12),
          if (posts.isEmpty)
            _ExploreEmptyHint(
              isDark: isDark,
              textColor: textColor,
              hintColor: hintColor,
              hasTag: hasTag,
              onClearFilters: onClearFilters,
            )
          else
            Builder(builder: (context) {
              return ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: posts.length,
                separatorBuilder: (_, __) => const SizedBox(height: 10),
                itemBuilder: (context, index) {
                  final post = posts[index];
                  return InkWell(
                    borderRadius: BDDesign.radiusLarge,
                    onTap: () => onTapPost(post),
                    child: BDPanelCard(
                      padding: const EdgeInsets.all(14),
                      child: Row(
                        children: [
                          _CommunityThumbnail(
                            imageUrl: post.coverUrl,
                            height: 118,
                            width: 104,
                            icon: Icons.terrain_rounded,
                          ),
                          const SizedBox(width: 14),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  post.placeName,
                                  style: TextStyle(
                                    color: textColor,
                                    fontSize: 18,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  '${post.latitude.toStringAsFixed(3)}, ${post.longitude.toStringAsFixed(3)}',
                                  style: TextStyle(
                                    color: hintColor,
                                    fontSize: 12.5,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  post.title,
                                  maxLines: 2,
                                  overflow: TextOverflow.ellipsis,
                                  style: TextStyle(
                                    color: textColor,
                                    fontWeight: FontWeight.w700,
                                    height: 1.2,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                BDStatusPill(
                                  label: post.modelName,
                                  icon: Icons.view_in_ar_rounded,
                                  color: BDDesign.colorMutedBlue,
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                },
              );
            }),
        ],
      ),
    );
  }
}

// ============================================================
// Discover Tab — 双列网格 + 标签筛选
// ============================================================

class CommunityDiscoverView extends StatelessWidget {
  final List<CommunityPost> posts;
  final Set<String> selectedTags;
  final ValueChanged<String> onToggleTag;
  final ValueChanged<CommunityPost> onTapPost;

  static const presetTags = [
    '街景', '建筑', '自然', '室内', '夜景',
    '人物', '美食', '旅行', '城市', '水景',
  ];

  const CommunityDiscoverView({
    super.key,
    required this.posts,
    required this.selectedTags,
    required this.onToggleTag,
    required this.onTapPost,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.55)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.72);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 4, 20, 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text(
                    textLocalize('community_discover_filter'),
                    style: TextStyle(
                        color: textColor,
                        fontSize: 14,
                        fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    '${posts.length}${textLocalize('community_discover_results')}',
                    style: TextStyle(color: hintColor, fontSize: 12),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: presetTags.map((tag) {
                  final selected = selectedTags.contains(tag);
                  return GestureDetector(
                    onTap: () => onToggleTag(tag),
                    child: AnimatedContainer(
                      duration: BDMotion.durationFast,
                      curve: BDMotion.curveFluid,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 8),
                      decoration: BoxDecoration(
                        color: selected
                            ? const Color(0xFF2E7CF6)
                            : (isDark
                                ? AppTheme.darkSurfaceElevated
                                : const Color(0xFFF3F5F9)),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                          color: selected
                              ? const Color(0xFF2E7CF6)
                              : (isDark
                                  ? Colors.white.withValues(alpha: 0.08)
                                  : BDDesign.colorMutedBlue
                                      .withValues(alpha: 0.12)),
                        ),
                      ),
                      child: Text(
                        tag,
                        style: TextStyle(
                          color: selected ? Colors.white : textColor,
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  );
                }).toList(),
              ),
            ],
          ),
        ),
        Expanded(
          child: posts.isEmpty
              ? const _CommunityEmptyState()
              : GridView.builder(
                  padding: const EdgeInsets.fromLTRB(16, 4, 16, 104),
                  gridDelegate:
                      const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    mainAxisSpacing: 12,
                    crossAxisSpacing: 12,
                    childAspectRatio: 0.72,
                  ),
                  itemCount: posts.length,
                  itemBuilder: (context, index) {
                    final post = posts[index];
                    return _DiscoverCard(
                      post: post,
                      onTap: () => onTapPost(post),
                    );
                  },
                ),
        ),
      ],
    );
  }
}

class _DiscoverCard extends StatelessWidget {
  final CommunityPost post;
  final VoidCallback onTap;

  const _DiscoverCard({required this.post, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.55)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.72);

    return GestureDetector(
      onTap: onTap,
      child: BDPanelCard(
        padding: EdgeInsets.zero,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Thumbnail
            ClipRRect(
              borderRadius: const BorderRadius.vertical(
                  top: Radius.circular(22)),
              child: _CommunityThumbnail(
                imageUrl: post.coverUrl,
                height: 140,
                width: double.infinity,
                icon: Icons.terrain_rounded,
              ),
            ),
            // Title + author
            Expanded(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      post.title,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: textColor,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        height: 1.25,
                      ),
                    ),
                    const Spacer(),
                    Text(
                      post.authorName,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                          color: hintColor,
                          fontSize: 12,
                          fontWeight: FontWeight.w500),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ============================================================
// Submit Tab — 多模型选择 + 草稿 + 发布
// ============================================================

class CommunitySubmitView extends StatelessWidget {
  final List<CommunityModelOption> shareableModels;
  final List<CommunityModelOption> selectedModels;
  final ValueChanged<CommunityModelOption> onToggleModel;
  final TextEditingController titleController;
  final TextEditingController captionController;
  final TextEditingController placeController;
  final TextEditingController latController;
  final TextEditingController lngController;
  final bool isSubmitting;
  final VoidCallback onSubmit;
  final VoidCallback onSaveDraft;

  static const _presets = <_LocationPreset>[
    _LocationPreset('西湖', 30.243, 120.150),
    _LocationPreset('外滩', 31.240, 121.490),
    _LocationPreset('东京塔', 35.659, 139.745),
    _LocationPreset('巴黎左岸', 48.853, 2.349),
    _LocationPreset('纽约中央公园', 40.782, -73.965),
  ];

  const CommunitySubmitView({
    super.key,
    required this.shareableModels,
    required this.selectedModels,
    required this.onToggleModel,
    required this.titleController,
    required this.captionController,
    required this.placeController,
    required this.latController,
    required this.lngController,
    required this.isSubmitting,
    required this.onSubmit,
    required this.onSaveDraft,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    final inputBorder = OutlineInputBorder(
      borderRadius: BDDesign.radiusLarge,
      borderSide: BorderSide.none,
    );
    final inputFill =
        isDark ? AppTheme.darkSurfaceElevated : const Color(0xFFF7FAFD);

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 4, 20, 104),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 模型选择区
          Text(
            textLocalize('community_select_model_label'),
            style: TextStyle(
                color: textColor,
                fontSize: 14,
                fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          if (shareableModels.isEmpty)
            BDPanelCard(
              padding: const EdgeInsets.all(20),
              child: Text(
                '还没有可分享的模型，请先在创作页面生成记忆模型。',
                style: TextStyle(color: hintColor, height: 1.4),
              ),
            )
          else
            BDPanelCard(
              padding: const EdgeInsets.all(10),
              child: Column(
                children: shareableModels.map((model) {
                  final isSelected =
                      selectedModels.any((m) => m.id == model.id);
                  return GestureDetector(
                    onTap: () => onToggleModel(model),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 10),
                      margin: const EdgeInsets.only(bottom: 4),
                      decoration: BoxDecoration(
                        color: isSelected
                            ? const Color(0xFF2E7CF6)
                                .withValues(alpha: 0.10)
                            : Colors.transparent,
                        borderRadius: BDDesign.radiusLarge,
                        border: Border.all(
                          color: isSelected
                              ? const Color(0xFF2E7CF6)
                              : Colors.transparent,
                        ),
                      ),
                      child: Row(
                        children: [
                          _CommunityThumbnail(
                            imageUrl: model.coverUrl,
                            height: 40,
                            width: 40,
                            icon: Icons.view_in_ar_rounded,
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment:
                                  CrossAxisAlignment.start,
                              children: [
                                Text(model.sceneId,
                                    style: TextStyle(
                                        color: textColor,
                                        fontWeight: FontWeight.w600)),
                                if (model.description.isNotEmpty)
                                  Text(model.description,
                                      maxLines: 1,
                                      overflow:
                                          TextOverflow.ellipsis,
                                      style: TextStyle(
                                          color: hintColor,
                                          fontSize: 12)),
                              ],
                            ),
                          ),
                          if (isSelected)
                            const Icon(
                              Icons.check_circle_rounded,
                              color: Color(0xFF2E7CF6),
                              size: 22,
                            ),
                        ],
                      ),
                    ),
                  );
                }).toList(),
              ),
            ),
          const SizedBox(height: 16),
          // 标题
          TextField(
            controller: titleController,
            decoration: InputDecoration(
              labelText: textLocalize('community_input_title'),
              filled: true,
              fillColor: inputFill,
              border: inputBorder,
            ),
          ),
          const SizedBox(height: 12),
          // 简介
          TextField(
            controller: captionController,
            minLines: 3,
            maxLines: 5,
            decoration: InputDecoration(
              labelText: textLocalize('community_input_caption'),
              filled: true,
              fillColor: inputFill,
              border: inputBorder,
            ),
          ),
          const SizedBox(height: 14),
          // 地点预设
          Text(
            textLocalize('community_location_preset'),
            style: TextStyle(
                color: textColor, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _presets.map((preset) {
              return ActionChip(
                label: Text(preset.name),
                onPressed: () {
                  placeController.text = preset.name;
                  latController.text =
                      preset.latitude.toStringAsFixed(3);
                  lngController.text =
                      preset.longitude.toStringAsFixed(3);
                },
              );
            }).toList(),
          ),
          const SizedBox(height: 12),
          // 地点名
          TextField(
            controller: placeController,
            decoration: InputDecoration(
              labelText: textLocalize('community_input_place'),
              filled: true,
              fillColor: inputFill,
              border: inputBorder,
            ),
          ),
          const SizedBox(height: 12),
          // 经纬度
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: latController,
                  keyboardType: const TextInputType.numberWithOptions(
                      decimal: true, signed: true),
                  decoration: InputDecoration(
                    labelText: textLocalize('community_input_lat'),
                    filled: true,
                    fillColor: inputFill,
                    border: inputBorder,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: TextField(
                  controller: lngController,
                  keyboardType: const TextInputType.numberWithOptions(
                      decimal: true, signed: true),
                  decoration: InputDecoration(
                    labelText: textLocalize('community_input_lng'),
                    filled: true,
                    fillColor: inputFill,
                    border: inputBorder,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          // 存草稿 + 发布按钮
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: onSaveDraft,
                  icon: const Icon(Icons.save_outlined),
                  label: const Text('存草稿'),
                  style: OutlinedButton.styleFrom(
                    padding:
                        const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BDDesign.radiusLarge,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                flex: 2,
                child: FilledButton.icon(
                  onPressed: (selectedModels.isEmpty || isSubmitting)
                      ? null
                      : onSubmit,
                  icon: isSubmitting
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(
                              strokeWidth: 2),
                        )
                      : const Icon(Icons.send_rounded),
                  label: Text(
                    isSubmitting
                        ? textLocalize('community_publishing')
                        : textLocalize('community_publish'),
                  ),
                  style: FilledButton.styleFrom(
                    padding:
                        const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BDDesign.radiusLarge,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ============================================================
// Shared Widgets
// ============================================================

class CommunityLocationHubRow extends StatelessWidget {
  final CommunityPost post;

  const CommunityLocationHubRow({super.key, required this.post});

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: isDark
            ? AppTheme.darkSurfaceElevated.withValues(alpha: 0.94)
            : const Color(0xFFF6F8FB),
        borderRadius: BDDesign.radiusLarge,
        border: Border.all(
          color: isDark
              ? Colors.white.withValues(alpha: 0.06)
              : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
        ),
      ),
      child: Row(
        children: [
          _CommunityThumbnail(
            imageUrl: post.coverUrl,
            height: 72,
            width: 92,
            icon: Icons.landscape_rounded,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(post.title,
                    style: TextStyle(
                        color: textColor,
                        fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                Text('${post.authorName} · ${post.modelName}',
                    style: TextStyle(
                        color: hintColor, fontSize: 12.5)),
                const SizedBox(height: 8),
                Text(post.caption,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                        color: textColor.withValues(alpha: 0.82),
                        height: 1.3)),
              ],
            ),
          ),
          const SizedBox(width: 8),
          Icon(Icons.chevron_right_rounded, color: hintColor),
        ],
      ),
    );
  }
}

class CommunityMetricCard extends StatelessWidget {
  final String label;
  final String value;
  final String hint;

  const CommunityMetricCard({
    super.key,
    required this.label,
    required this.value,
    required this.hint,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    return BDPanelCard(
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: TextStyle(
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.58)
                      : BDDesign.colorMutedBlue,
                  fontWeight: FontWeight.w700,
                  fontSize: 12.5)),
          const SizedBox(height: 8),
          Text(value,
              style: TextStyle(
                  color: isDark
                      ? BDDesign.colorPaperWhite
                      : BDDesign.colorInkBlack,
                  fontSize: 24,
                  fontWeight: FontWeight.w800)),
          const SizedBox(height: 6),
          Text(hint,
              style: TextStyle(
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.54)
                      : BDDesign.colorMutedBlue
                          .withValues(alpha: 0.86),
                  height: 1.35,
                  fontSize: 12.5)),
        ],
      ),
    );
  }
}

// ------- Private Helpers -------

class _CommunityThumbnail extends StatelessWidget {
  final String? imageUrl;
  final double height;
  final double width;
  final IconData icon;

  const _CommunityThumbnail({
    required this.imageUrl,
    required this.height,
    required this.width,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    final fallback = Container(
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFF8BA8C5),
            Color(0xFF536C8B),
            Color(0xFF38485F),
          ],
        ),
        borderRadius: BorderRadius.circular(22),
      ),
      child: Center(
        child: Icon(
          icon,
          size: 34,
          color: Colors.white.withValues(alpha: 0.88),
        ),
      ),
    );

    final url = imageUrl;
    return SizedBox(
      height: height,
      width: width,
      child: url == null || url.isEmpty
          ? ClipRRect(
              borderRadius: BorderRadius.circular(22),
              child: fallback,
            )
          : BDFadeInNetworkImage(
              imageUrl: url,
              placeholder: fallback,
              errorWidget: ClipRRect(
                borderRadius: BorderRadius.circular(22),
                child: fallback,
              ),
              fit: BoxFit.cover,
              borderRadius: BorderRadius.circular(22),
              backgroundColor: Colors.transparent,
              duration: BDMotion.durationSlow,
              curve: BDMotion.curveEnter,
            ),
    );
  }
}

class _CommunityEmptyState extends StatelessWidget {
  const _CommunityEmptyState();

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 6, 20, 104),
      child: BDPanelCard(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.public_off_rounded,
                size: 48,
                color: isDark
                    ? Colors.white.withValues(alpha: 0.48)
                    : BDDesign.colorMutedBlue,
              ),
              const SizedBox(height: 14),
              Text(
                textLocalize('community_empty_feed'),
                style: TextStyle(
                  color: isDark
                      ? BDDesign.colorPaperWhite
                      : BDDesign.colorInkBlack,
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                textLocalize('community_empty_hint'),
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.62)
                      : BDDesign.colorMutedBlue
                          .withValues(alpha: 0.88),
                  height: 1.4,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _LocationPreset {
  final String name;
  final double latitude;
  final double longitude;

  const _LocationPreset(this.name, this.latitude, this.longitude);
}

// ============================================================
// Explore filter bar + empty hint
// ============================================================

class _ExploreFilterBar extends StatelessWidget {
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final bool boundsReady;
  final int zoom;
  final double tagRadiusKm;
  final int visibleCount;
  final int totalCount;
  final int filteredCount;
  final List<String> availableTags;
  final String? selectedTag;
  final ValueChanged<String> onToggleTag;
  final VoidCallback onClearFilters;

  const _ExploreFilterBar({
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.boundsReady,
    required this.zoom,
    required this.tagRadiusKm,
    required this.visibleCount,
    required this.totalCount,
    required this.filteredCount,
    required this.availableTags,
    required this.selectedTag,
    required this.onToggleTag,
    required this.onClearFilters,
  });

  String _radiusLabel() {
    if (tagRadiusKm >= 1) {
      return '${tagRadiusKm.toStringAsFixed(tagRadiusKm >= 10 ? 0 : 1)} km';
    }
    return '${(tagRadiusKm * 1000).round()} m';
  }

  @override
  Widget build(BuildContext context) {
    final hasTag = selectedTag != null && selectedTag!.isNotEmpty;
    final summary = boundsReady
        ? (hasTag
            ? '当前区域 · 含 “$selectedTag” · ${_radiusLabel()} 内 $filteredCount 条'
            : '当前区域 $visibleCount/$totalCount 条 · ZOOM $zoom')
        : '调整地图后将按可视区域筛选 · 共 $totalCount 条';
    return BDPanelCard(
      padding: const EdgeInsets.fromLTRB(14, 12, 10, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.travel_explore_rounded,
                size: 18,
                color: hintColor,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  summary,
                  style: TextStyle(
                    color: textColor,
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    height: 1.3,
                  ),
                ),
              ),
              if (hasTag)
                TextButton.icon(
                  onPressed: onClearFilters,
                  icon: const Icon(Icons.close_rounded, size: 16),
                  label: const Text('清除'),
                  style: TextButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 4),
                    minimumSize: const Size(0, 32),
                    visualDensity: VisualDensity.compact,
                  ),
                ),
            ],
          ),
          if (availableTags.isNotEmpty) ...[
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: availableTags.map((tag) {
                final selected = tag == selectedTag;
                return _TagPill(
                  label: tag,
                  selected: selected,
                  isDark: isDark,
                  onTap: () => onToggleTag(tag),
                );
              }).toList(),
            ),
          ],
        ],
      ),
    );
  }
}

class _TagPill extends StatelessWidget {
  final String label;
  final bool selected;
  final bool isDark;
  final VoidCallback onTap;

  const _TagPill({
    required this.label,
    required this.selected,
    required this.isDark,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final textColor = selected
        ? Colors.white
        : (isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack);
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: AnimatedContainer(
        duration: BDMotion.durationFast,
        curve: BDMotion.curveFluid,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: selected
              ? const Color(0xFF2E7CF6)
              : (isDark
                  ? AppTheme.darkSurfaceElevated
                  : const Color(0xFFF3F5F9)),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
            color: selected
                ? const Color(0xFF2E7CF6)
                : (isDark
                    ? Colors.white.withValues(alpha: 0.08)
                    : BDDesign.colorMutedBlue.withValues(alpha: 0.12)),
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: textColor,
            fontSize: 12.5,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}

class _ExploreEmptyHint extends StatelessWidget {
  final bool isDark;
  final Color textColor;
  final Color hintColor;
  final bool hasTag;
  final VoidCallback onClearFilters;

  const _ExploreEmptyHint({
    required this.isDark,
    required this.textColor,
    required this.hintColor,
    required this.hasTag,
    required this.onClearFilters,
  });

  @override
  Widget build(BuildContext context) {
    return BDPanelCard(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                hasTag
                    ? Icons.filter_alt_off_rounded
                    : Icons.public_off_rounded,
                color: hintColor,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  hasTag ? '当前区域内没有匹配该标签的帖子' : '当前区域里还没有空间记忆',
                  style: TextStyle(
                    color: textColor,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            hasTag ? '试试切换其它标签，或清除筛选查看区域内全部帖子。' : '滑动或缩放地图，看看其它区域。',
            style: TextStyle(color: hintColor, height: 1.4, fontSize: 12.5),
          ),
          if (hasTag) ...[
            const SizedBox(height: 10),
            Align(
              alignment: Alignment.centerLeft,
              child: TextButton.icon(
                onPressed: onClearFilters,
                icon: const Icon(Icons.close_rounded, size: 16),
                label: const Text('清除筛选'),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
