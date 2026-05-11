import 'dart:math' as math;

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';

import 'models.dart';

// ============================================================
// Explore Tab — 地图探索
// ============================================================

class CommunityExploreView extends StatelessWidget {
  final List<CommunityPost> posts;
  final int selectedIndex;
  final ValueChanged<int> onSelect;
  final ValueChanged<CommunityPost> onOpenViewer;
  final ValueChanged<CommunityPost> onOpenLocationHub;
  final CommunityPost? selectedPost;

  const CommunityExploreView({
    super.key,
    required this.posts,
    required this.selectedIndex,
    required this.onSelect,
    required this.onOpenViewer,
    required this.onOpenLocationHub,
    required this.selectedPost,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    if (posts.isEmpty) {
      return const _CommunityEmptyState();
    }

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
                      label: '${posts.length} PINS',
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
                    return SizedBox(
                      height: mapHeight,
                      child: Stack(
                        children: [
                          Positioned.fill(
                            child: DecoratedBox(
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(26),
                                gradient: LinearGradient(
                                  begin: Alignment.topLeft,
                                  end: Alignment.bottomRight,
                                  colors: isDark
                                      ? const [
                                          Color(0xFF11161C),
                                          Color(0xFF1D2833),
                                          Color(0xFF11161C),
                                        ]
                                      : const [
                                          Color(0xFFEAF1F8),
                                          Color(0xFFDDE8F2),
                                          Color(0xFFF7FBFF),
                                        ],
                                ),
                              ),
                              child: CustomPaint(
                                painter: _WorldMapPainter(isDark: isDark),
                              ),
                            ),
                          ),
                          ...posts.asMap().entries.map((entry) {
                            final index = entry.key;
                            final post = entry.value;
                            final isSelected = index == selectedIndex;
                            final offset = _projectPoint(
                              post.latitude,
                              post.longitude,
                              mapWidth,
                              mapHeight,
                            );
                            return Positioned(
                              left: offset.dx - 16,
                              top: offset.dy - 16,
                              child: GestureDetector(
                                onTap: () => onSelect(index),
                                child: AnimatedContainer(
                                  duration: BDMotion.durationNormal,
                                  curve: BDMotion.curveFluid,
                                  width: isSelected ? 34 : 28,
                                  height: isSelected ? 34 : 28,
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    color: isSelected
                                        ? const Color(0xFFE9654B)
                                        : const Color(0xFF2E7CF6),
                                    border: Border.all(
                                      color: Colors.white.withValues(
                                        alpha: 0.92,
                                      ),
                                      width: 3,
                                    ),
                                  ),
                                  child: Icon(
                                    Icons.location_on_rounded,
                                    color: Colors.white,
                                    size: isSelected ? 18 : 16,
                                  ),
                                ),
                              ),
                            );
                          }),
                        ],
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          if (selectedPost != null)
            _SelectedLocationCard(
              post: selectedPost!,
              relatedCount: posts
                  .where((p) => p.placeName == selectedPost!.placeName)
                  .length,
              onOpenViewer: () => onOpenViewer(selectedPost!),
              onOpenLocationHub: () => onOpenLocationHub(selectedPost!),
            ),
          const SizedBox(height: 14),
          SizedBox(
            height: 154,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: posts.length,
              separatorBuilder: (_, __) => const SizedBox(width: 12),
              itemBuilder: (context, index) {
                final post = posts[index];
                final isActive = index == selectedIndex;
                return SizedBox(
                  width: 244,
                  child: InkWell(
                    borderRadius: BDDesign.radiusLarge,
                    onTap: () => onSelect(index),
                    child: AnimatedContainer(
                      duration: BDMotion.durationNormal,
                      curve: BDMotion.curveFluid,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: isActive
                            ? (isDark
                                ? AppTheme.darkSurfaceElevated
                                : const Color(0xFFF7FAFD))
                            : (isDark
                                ? AppTheme.darkSurface.withValues(alpha: 0.94)
                                : Colors.white.withValues(alpha: 0.88)),
                        borderRadius: BDDesign.radiusLarge,
                        border: Border.all(
                          color: isActive
                              ? const Color(0xFF2E7CF6)
                              : (isDark
                                  ? Colors.white.withValues(alpha: 0.06)
                                  : BDDesign.colorMutedBlue.withValues(
                                      alpha: 0.08,
                                    )),
                        ),
                      ),
                      child: Row(
                        children: [
                          _CommunityThumbnail(
                            imageUrl: post.coverUrl,
                            height: 130,
                            width: 88,
                            icon: Icons.explore_rounded,
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  post.placeName,
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: TextStyle(
                                    color: textColor,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  post.title,
                                  maxLines: 2,
                                  overflow: TextOverflow.ellipsis,
                                  style: TextStyle(
                                    color:
                                        textColor.withValues(alpha: 0.86),
                                    height: 1.25,
                                  ),
                                ),
                                const Spacer(),
                                BDStatusPill(
                                  label: post.modelName,
                                  icon: Icons.view_in_ar_rounded,
                                  color: isActive
                                      ? const Color(0xFF2E7CF6)
                                      : BDDesign.colorMutedBlue,
                                ),
                              ],
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
    );
  }
}

// ============================================================
// Discover Tab — 标签筛选 + 模型卡片列表
// ============================================================

class CommunityDiscoverView extends StatelessWidget {
  final List<CommunityPost> posts;
  final Set<String> selectedTags;
  final ValueChanged<String> onToggleTag;
  final ValueChanged<CommunityPost> onOpenViewer;

  static const presetTags = [
    '街景',
    '建筑',
    '自然',
    '室内',
    '夜景',
    '人物',
    '美食',
    '旅行',
    '城市',
    '水景',
  ];

  const CommunityDiscoverView({
    super.key,
    required this.posts,
    required this.selectedTags,
    required this.onToggleTag,
    required this.onOpenViewer,
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
                      fontWeight: FontWeight.w600,
                    ),
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
                        horizontal: 14,
                        vertical: 8,
                      ),
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
              : ListView.separated(
                  padding: const EdgeInsets.fromLTRB(20, 4, 20, 104),
                  itemCount: posts.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 10),
                  itemBuilder: (context, index) {
                    final post = posts[index];
                    return _CommunityDiscoverCard(
                      post: post,
                      onOpenViewer: () => onOpenViewer(post),
                    );
                  },
                ),
        ),
      ],
    );
  }
}

class _CommunityDiscoverCard extends StatelessWidget {
  final CommunityPost post;
  final VoidCallback onOpenViewer;

  const _CommunityDiscoverCard({
    required this.post,
    required this.onOpenViewer,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.55)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.72);

    return BDPanelCard(
      padding: const EdgeInsets.all(14),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _CommunityThumbnail(
            imageUrl: post.coverUrl,
            height: 96,
            width: 96,
            icon: Icons.terrain_rounded,
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  post.title,
                  style: TextStyle(
                    color: textColor,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${post.authorName} · ${post.placeName}',
                  style: TextStyle(color: hintColor, fontSize: 12.5),
                ),
                const SizedBox(height: 8),
                Text(
                  post.caption,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: textColor.withValues(alpha: 0.78),
                    height: 1.35,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(height: 10),
                Wrap(
                  crossAxisAlignment: WrapCrossAlignment.center,
                  spacing: 6,
                  runSpacing: 4,
                  children: [
                    ...post.tags.take(4).map(
                          (tag) => _TagChip(label: tag, isDark: isDark),
                        ),
                    TextButton.icon(
                      onPressed: onOpenViewer,
                      icon: const Icon(
                        Icons.play_circle_fill_rounded,
                        size: 18,
                      ),
                      label: Text(
                        textLocalize('community_enter_memory'),
                        style: const TextStyle(fontSize: 13),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _TagChip extends StatelessWidget {
  final String label;
  final bool isDark;

  const _TagChip({required this.label, required this.isDark});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: isDark
            ? Colors.white.withValues(alpha: 0.06)
            : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: BDDesign.colorMutedBlue,
          fontSize: 11,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

// ============================================================
// Submit Tab — 发布表单
// ============================================================

class CommunitySubmitView extends StatelessWidget {
  final CommunityModelOption? selectedModel;
  final VoidCallback onPickModel;
  final TextEditingController titleController;
  final TextEditingController captionController;
  final TextEditingController placeController;
  final TextEditingController latController;
  final TextEditingController lngController;
  final bool isSubmitting;
  final VoidCallback onSubmit;

  static const _presets = <_LocationPreset>[
    _LocationPreset('西湖', 30.243, 120.150),
    _LocationPreset('外滩', 31.240, 121.490),
    _LocationPreset('东京塔', 35.659, 139.745),
    _LocationPreset('巴黎左岸', 48.853, 2.349),
    _LocationPreset('纽约中央公园', 40.782, -73.965),
  ];

  const CommunitySubmitView({
    super.key,
    required this.selectedModel,
    required this.onPickModel,
    required this.titleController,
    required this.captionController,
    required this.placeController,
    required this.latController,
    required this.lngController,
    required this.isSubmitting,
    required this.onSubmit,
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
          // 模型选择器
          Text(
            textLocalize('community_select_model_label'),
            style: TextStyle(
              color: textColor,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          GestureDetector(
            onTap: onPickModel,
            child: BDPanelCard(
              padding: const EdgeInsets.all(14),
              child: Row(
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: isDark
                          ? AppTheme.darkSurfaceElevated
                          : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
                      borderRadius: BDDesign.radiusSmall,
                    ),
                    child: Icon(
                      selectedModel != null
                          ? Icons.view_in_ar_rounded
                          : Icons.add_photo_alternate_rounded,
                      color: hintColor,
                      size: 24,
                    ),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          selectedModel?.sceneId ??
                              textLocalize('community_select_model_hint'),
                          style: TextStyle(
                            color:
                                selectedModel != null ? textColor : hintColor,
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        if (selectedModel != null) ...[
                          const SizedBox(height: 2),
                          Text(
                            selectedModel!.description,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style:
                                TextStyle(color: hintColor, fontSize: 12),
                          ),
                        ],
                      ],
                    ),
                  ),
                  Icon(Icons.chevron_right_rounded, color: hintColor),
                ],
              ),
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
          // 快速地点预设
          Text(
            textLocalize('community_location_preset'),
            style: TextStyle(
              color: textColor,
              fontWeight: FontWeight.w600,
            ),
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
                    decimal: true,
                    signed: true,
                  ),
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
                    decimal: true,
                    signed: true,
                  ),
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
          // 发布按钮
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed:
                  (selectedModel == null || isSubmitting) ? null : onSubmit,
              icon: isSubmitting
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.send_rounded),
              label: Text(
                isSubmitting
                    ? textLocalize('community_publishing')
                    : textLocalize('community_publish'),
              ),
              style: FilledButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BDDesign.radiusLarge,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// Model Picker Bottom Sheet
// ============================================================

class CommunityModelPickerSheet extends StatefulWidget {
  final List<CommunityModelOption> models;
  final CommunityModelOption? selectedModel;

  const CommunityModelPickerSheet({
    super.key,
    required this.models,
    this.selectedModel,
  });

  @override
  State<CommunityModelPickerSheet> createState() =>
      _CommunityModelPickerSheetState();
}

class _CommunityModelPickerSheetState
    extends State<CommunityModelPickerSheet> {
  String _query = '';
  final TextEditingController _searchController = TextEditingController();

  List<CommunityModelOption> get _filtered {
    if (_query.isEmpty) return widget.models;
    final q = _query.toLowerCase();
    return widget.models
        .where((m) =>
            m.sceneId.toLowerCase().contains(q) ||
            m.description.toLowerCase().contains(q))
        .toList();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);
    final models = _filtered;

    return Padding(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
      ),
      child: DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.7,
        maxChildSize: 0.9,
        minChildSize: 0.4,
        builder: (context, scrollController) {
          return Padding(
            padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
            child: BDPanelCard(
              padding: const EdgeInsets.fromLTRB(18, 18, 18, 12),
              child: SafeArea(
                top: false,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            textLocalize('community_model_picker_title'),
                            style: TextStyle(
                              color: textColor,
                              fontSize: 20,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                        IconButton(
                          onPressed: () => Navigator.pop(context),
                          icon:
                              Icon(Icons.close_rounded, color: textColor),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _searchController,
                      onChanged: (v) => setState(() => _query = v),
                      decoration: InputDecoration(
                        hintText: textLocalize('community_search_model'),
                        prefixIcon: const Icon(Icons.search_rounded),
                        filled: true,
                        fillColor: isDark
                            ? AppTheme.darkSurfaceElevated
                            : const Color(0xFFF7FAFD),
                        border: OutlineInputBorder(
                          borderRadius: BDDesign.radiusLarge,
                          borderSide: BorderSide.none,
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    Flexible(
                      child: models.isEmpty
                          ? Center(
                              child: Text(
                                textLocalize('community_no_model_found'),
                                style: TextStyle(color: hintColor),
                              ),
                            )
                          : ListView.separated(
                              controller: scrollController,
                              shrinkWrap: true,
                              itemCount: models.length,
                              separatorBuilder: (_, __) =>
                                  const SizedBox(height: 8),
                              itemBuilder: (context, index) {
                                final model = models[index];
                                final isSelected =
                                    widget.selectedModel?.id == model.id;
                                return InkWell(
                                  onTap: () =>
                                      Navigator.pop(context, model),
                                  borderRadius: BDDesign.radiusLarge,
                                  child: Container(
                                    padding: const EdgeInsets.all(12),
                                    decoration: BoxDecoration(
                                      color: isSelected
                                          ? const Color(0xFF2E7CF6)
                                              .withValues(alpha: 0.12)
                                          : Colors.transparent,
                                      borderRadius: BDDesign.radiusLarge,
                                      border: Border.all(
                                        color: isSelected
                                            ? const Color(0xFF2E7CF6)
                                            : (isDark
                                                ? Colors.white
                                                    .withValues(alpha: 0.06)
                                                : Colors.transparent),
                                      ),
                                    ),
                                    child: Row(
                                      children: [
                                        _CommunityThumbnail(
                                          imageUrl: model.coverUrl,
                                          height: 44,
                                          width: 44,
                                          icon: Icons.view_in_ar_rounded,
                                        ),
                                        const SizedBox(width: 12),
                                        Expanded(
                                          child: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                model.sceneId,
                                                style: TextStyle(
                                                  color: textColor,
                                                  fontWeight: FontWeight.w600,
                                                ),
                                              ),
                                              if (model.description
                                                  .isNotEmpty) ...[
                                                const SizedBox(height: 2),
                                                Text(
                                                  model.description,
                                                  maxLines: 1,
                                                  overflow:
                                                      TextOverflow.ellipsis,
                                                  style: TextStyle(
                                                    color: hintColor,
                                                    fontSize: 12,
                                                  ),
                                                ),
                                              ],
                                            ],
                                          ),
                                        ),
                                        if (isSelected)
                                          const Icon(
                                            Icons.check_circle_rounded,
                                            color: Color(0xFF2E7CF6),
                                          ),
                                      ],
                                    ),
                                  ),
                                );
                              },
                            ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
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
                Text(
                  post.title,
                  style: TextStyle(
                    color: textColor,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${post.authorName} · ${post.modelName}',
                  style: TextStyle(color: hintColor, fontSize: 12.5),
                ),
                const SizedBox(height: 8),
                Text(
                  post.caption,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: textColor.withValues(alpha: 0.82),
                    height: 1.3,
                  ),
                ),
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
          Text(
            label,
            style: TextStyle(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.58)
                  : BDDesign.colorMutedBlue,
              fontWeight: FontWeight.w700,
              fontSize: 12.5,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: TextStyle(
              color: isDark
                  ? BDDesign.colorPaperWhite
                  : BDDesign.colorInkBlack,
              fontSize: 24,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            hint,
            style: TextStyle(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.54)
                  : BDDesign.colorMutedBlue.withValues(alpha: 0.86),
              height: 1.35,
              fontSize: 12.5,
            ),
          ),
        ],
      ),
    );
  }
}

// ------- Private Helpers -------

class _SelectedLocationCard extends StatelessWidget {
  final CommunityPost post;
  final int relatedCount;
  final VoidCallback onOpenViewer;
  final VoidCallback onOpenLocationHub;

  const _SelectedLocationCard({
    required this.post,
    required this.relatedCount,
    required this.onOpenViewer,
    required this.onOpenLocationHub,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return BDPanelCard(
      padding: const EdgeInsets.all(14),
      child: Column(
        children: [
          Row(
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
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${post.latitude.toStringAsFixed(3)}, ${post.longitude.toStringAsFixed(3)}',
                      style: TextStyle(color: hintColor, fontSize: 12.5),
                    ),
                    const SizedBox(height: 10),
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
                    Text(
                      '这一地点已聚合 $relatedCount 条记忆，优先展示当前最热的 3D 模型。',
                      style: TextStyle(color: hintColor, height: 1.35),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: onOpenViewer,
                  icon: const Icon(Icons.travel_explore_rounded),
                  label: Text(textLocalize('community_open_model')),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: onOpenLocationHub,
                  icon: const Icon(Icons.map_rounded),
                  label: Text(textLocalize('community_view_location')),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

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
      child: ClipRRect(
        borderRadius: BorderRadius.circular(22),
        child: url == null || url.isEmpty
            ? fallback
            : Image.network(
                url,
                fit: BoxFit.cover,
                errorBuilder: (_, _, _) => fallback,
                loadingBuilder: (context, child, loadingProgress) {
                  if (loadingProgress == null) return child;
                  return fallback;
                },
              ),
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
                      : BDDesign.colorMutedBlue.withValues(alpha: 0.88),
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

class _WorldMapPainter extends CustomPainter {
  final bool isDark;

  const _WorldMapPainter({required this.isDark});

  @override
  void paint(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = (isDark ? Colors.white : BDDesign.colorMutedBlue)
          .withValues(alpha: 0.08)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;

    for (var i = 1; i < 6; i++) {
      final dy = size.height * i / 6;
      canvas.drawLine(Offset(0, dy), Offset(size.width, dy), gridPaint);
    }

    final landPaint = Paint()
      ..color = isDark
          ? const Color(0xFFAFD0E9).withValues(alpha: 0.16)
          : const Color(0xFF8AA9C7).withValues(alpha: 0.28);

    void drawLand(List<Offset> points) {
      final path = Path()
        ..moveTo(points.first.dx * size.width, points.first.dy * size.height);
      for (final point in points.skip(1)) {
        path.lineTo(point.dx * size.width, point.dy * size.height);
      }
      path.close();
      canvas.drawPath(path, landPaint);
    }

    drawLand(const [
      Offset(0.08, 0.20),
      Offset(0.20, 0.13),
      Offset(0.28, 0.17),
      Offset(0.32, 0.28),
      Offset(0.25, 0.40),
      Offset(0.18, 0.38),
      Offset(0.13, 0.46),
      Offset(0.09, 0.35),
    ]);
    drawLand(const [
      Offset(0.42, 0.18),
      Offset(0.52, 0.13),
      Offset(0.62, 0.18),
      Offset(0.67, 0.30),
      Offset(0.61, 0.36),
      Offset(0.54, 0.32),
      Offset(0.49, 0.35),
      Offset(0.46, 0.26),
    ]);
    drawLand(const [
      Offset(0.67, 0.23),
      Offset(0.82, 0.19),
      Offset(0.92, 0.28),
      Offset(0.88, 0.42),
      Offset(0.78, 0.43),
      Offset(0.71, 0.36),
    ]);
  }

  @override
  bool shouldRepaint(covariant _WorldMapPainter oldDelegate) {
    return oldDelegate.isDark != isDark;
  }
}

Offset _projectPoint(double lat, double lng, double w, double h) {
  return Offset(
    ((lng + 180) / 360 * w).clamp(16.0, w - 16.0),
    ((90 - lat) / 180 * h).clamp(16.0, h - 16.0),
  );
}

class _LocationPreset {
  final String name;
  final double latitude;
  final double longitude;

  const _LocationPreset(this.name, this.latitude, this.longitude);
}
