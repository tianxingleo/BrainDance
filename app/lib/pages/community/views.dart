import 'dart:math' as math;

import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:braindance/widgets/animated_network_image.dart';
import 'package:flutter/material.dart';

import 'models.dart';

class CommunityFeedView extends StatelessWidget {
  final List<CommunityPost> posts;
  final PageController controller;
  final ValueChanged<CommunityPost> onOpenViewer;
  final ValueChanged<CommunityPost> onOpenLocationHub;

  const CommunityFeedView({
    super.key,
    required this.posts,
    required this.controller,
    required this.onOpenViewer,
    required this.onOpenLocationHub,
  });

  @override
  Widget build(BuildContext context) {
    if (posts.isEmpty) {
      return const _CommunityEmptyState();
    }

    return PageView.builder(
      controller: controller,
      scrollDirection: Axis.vertical,
      itemCount: posts.length,
      padEnds: false,
      itemBuilder: (context, index) {
        final post = posts[index];
        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 104),
          child: _CommunityFeedCard(
            post: post,
            onOpenViewer: () => onOpenViewer(post),
            onOpenLocationHub: () => onOpenLocationHub(post),
          ),
        );
      },
    );
  }
}

class CommunityMapView extends StatelessWidget {
  final List<CommunityPost> posts;
  final int selectedIndex;
  final ValueChanged<int> onSelect;
  final ValueChanged<CommunityPost> onOpenViewer;
  final ValueChanged<CommunityPost> onOpenLocationHub;
  final CommunityPost? selectedPost;

  const CommunityMapView({
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
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
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
                  .where((post) => post.placeName == selectedPost!.placeName)
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
              separatorBuilder: (_, _) => const SizedBox(width: 12),
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
                                    color: textColor.withValues(alpha: 0.86),
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

  Offset _projectPoint(
    double latitude,
    double longitude,
    double width,
    double height,
  ) {
    final dx = (longitude + 180) / 360 * width;
    final dy = (90 - latitude) / 180 * height;
    return Offset(dx.clamp(16.0, width - 16.0), dy.clamp(16.0, height - 16.0));
  }
}

class CommunityLocationHubRow extends StatelessWidget {
  final CommunityPost post;

  const CommunityLocationHubRow({super.key, required this.post});

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
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
              color: isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack,
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

class _CommunityFeedCard extends StatelessWidget {
  final CommunityPost post;
  final VoidCallback onOpenViewer;
  final VoidCallback onOpenLocationHub;

  const _CommunityFeedCard({
    required this.post,
    required this.onOpenViewer,
    required this.onOpenLocationHub,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.66)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.84);

    return BDPanelCard(
      borderRadius: BorderRadius.circular(30),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(30),
        child: Stack(
          fit: StackFit.expand,
          children: [
            _CommunityThumbnail(
              imageUrl: post.coverUrl,
              height: double.infinity,
              width: double.infinity,
              icon: Icons.slow_motion_video_rounded,
            ),
            DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.black.withValues(alpha: 0.08),
                    Colors.black.withValues(alpha: 0.18),
                    Colors.black.withValues(alpha: 0.74),
                  ],
                ),
              ),
            ),
            Positioned(
              top: 18,
              left: 18,
              right: 18,
              child: Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  BDStatusPill(
                    label: post.modelName,
                    icon: Icons.view_in_ar_rounded,
                    color: const Color(0xFF78A6FF),
                  ),
                  BDStatusPill(
                    label: post.placeName,
                    icon: Icons.place_rounded,
                    color: const Color(0xFFEAC86B),
                  ),
                ],
              ),
            ),
            Positioned(
              left: 18,
              right: 18,
              bottom: 18,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    post.title,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 28,
                      fontWeight: FontWeight.w800,
                      height: 1.05,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    post.caption,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.88),
                      height: 1.45,
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 14),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      _TranslucentTag(label: post.authorName),
                      _TranslucentTag(label: post.relativeTimeLabel),
                      ...post.tags
                          .take(3)
                          .map((tag) => _TranslucentTag(label: tag)),
                    ],
                  ),
                  const SizedBox(height: 18),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          onPressed: onOpenViewer,
                          icon: const Icon(Icons.play_circle_fill_rounded),
                          label: Text(textLocalize('community_enter_memory')),
                        ),
                      ),
                      const SizedBox(width: 10),
                      OutlinedButton.icon(
                        onPressed: onOpenLocationHub,
                        icon: Icon(
                          Icons.public_rounded,
                          color: isDark ? BDDesign.colorPaperWhite : textColor,
                        ),
                        label: Text(
                          textLocalize('community_location_content'),
                          style: TextStyle(
                            color: isDark
                                ? BDDesign.colorPaperWhite
                                : textColor,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '向下继续刷，可以切到下一个地点的 3D 记忆。',
                    style: TextStyle(color: hintColor, fontSize: 12.5),
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
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
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
          colors: [Color(0xFF8BA8C5), Color(0xFF536C8B), Color(0xFF38485F)],
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

class _TranslucentTag extends StatelessWidget {
  final String label;

  const _TranslucentTag({required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withValues(alpha: 0.18)),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: Colors.white.withValues(alpha: 0.88),
          fontWeight: FontWeight.w600,
          fontSize: 12.5,
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
      ..color = (isDark ? Colors.white : BDDesign.colorMutedBlue).withValues(
        alpha: 0.08,
      )
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
