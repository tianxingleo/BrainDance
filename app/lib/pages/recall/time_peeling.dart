/// 时间线剥离视图
///
/// 将模型按名称分组，每组显示一个水平时间线条带（carousel），
/// 带有连接节点的时间线和动画选中效果。
library;

import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../../configs/app_config.dart';
import '../../configs/motion_tokens.dart';
import 'model_grid.dart';
import 'model_grid_helpers.dart';

/// 时间线颜色（暖橙色）
const Color kTimelineColor = Color(0xFFCC9A5C);

/// 轮播区域高度
const double kCarouselHeight = 240.0;

/// 轮播视口比例
const double kViewportFraction = 0.52;

/// 边缘淡出宽度
const double kEdgeFadeWidth = 36.0;

/// 时间线剥离列表
///
/// 将模型按名称分组，每组一个带有水平时间线条带的 slot。
class TimePeelingList extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Map<String, List<Map<String, dynamic>>> groupedModels;
  final Map<String, dynamic>? activeModelAction;
  final GlobalKey Function(Map<String, dynamic>) modelCardKeyFor;
  final bool Function(Map<String, dynamic>?, Map<String, dynamic>?) isSameModel;
  final void Function(Map<String, dynamic>, dynamic) onNavigateToViewer;
  final void Function(Map<String, dynamic> model, {bool imageOnly})
  onShowModelActions;
  final void Function(String name) onAddNewTask;

  const TimePeelingList({
    super.key,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkInput,
    required this.groupedModels,
    required this.activeModelAction,
    required this.modelCardKeyFor,
    required this.isSameModel,
    required this.onNavigateToViewer,
    required this.onShowModelActions,
    required this.onAddNewTask,
  });

  @override
  Widget build(BuildContext context) {
    final textColor = resolveTextColor(isDark);
    final hintTextColor = resolveHintTextColor(isDark, theme);
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.55)
        : BDDesign.colorMutedBlue;

    final sortedKeys = groupedModels.keys.toList()
      ..sort((a, b) {
        final ta = _newestTime(groupedModels[a]!);
        final tb = _newestTime(groupedModels[b]!);
        return tb.compareTo(ta);
      });

    return SliverPadding(
      padding: const EdgeInsets.fromLTRB(0, 6, 0, 16),
      sliver: SliverList(
        delegate: SliverChildBuilderDelegate((context, index) {
          final name = sortedKeys[index];
          final models = groupedModels[name]!;
          return TimePeelingSlot(
            name: name,
            models: models,
            theme: theme,
            isDark: isDark,
            darkCard: darkCard,
            darkInput: darkInput,
            textColor: textColor,
            hintTextColor: hintTextColor,
            hintColor: hintColor,
            activeModelAction: activeModelAction,
            modelCardKeyFor: modelCardKeyFor,
            isSameModel: isSameModel,
            onNavigateToViewer: onNavigateToViewer,
            onShowModelActions: onShowModelActions,
            onAddNewTask: onAddNewTask,
          );
        }, childCount: sortedKeys.length),
      ),
    );
  }

  DateTime _newestTime(List<Map<String, dynamic>> models) {
    return models
        .map(
          (m) =>
              DateTime.tryParse(m['created_at']?.toString() ?? '') ??
              DateTime(0),
        )
        .reduce((a, b) => a.isAfter(b) ? a : b);
  }
}

/// 时间线单槽
///
/// 展示单个模型分组的轮播卡片和底部时间线节点。
class TimePeelingSlot extends StatefulWidget {
  final String name;
  final List<Map<String, dynamic>> models;
  final TDThemeData theme;
  final bool isDark;
  final Color darkCard;
  final Color darkInput;
  final Color textColor;
  final Color hintTextColor;
  final Color hintColor;
  final Map<String, dynamic>? activeModelAction;
  final GlobalKey Function(Map<String, dynamic>) modelCardKeyFor;
  final bool Function(Map<String, dynamic>?, Map<String, dynamic>?) isSameModel;
  final void Function(Map<String, dynamic>, dynamic) onNavigateToViewer;
  final void Function(Map<String, dynamic> model, {bool imageOnly})
  onShowModelActions;
  final void Function(String name) onAddNewTask;

  const TimePeelingSlot({
    super.key,
    required this.name,
    required this.models,
    required this.theme,
    required this.isDark,
    required this.darkCard,
    required this.darkInput,
    required this.textColor,
    required this.hintTextColor,
    required this.hintColor,
    required this.activeModelAction,
    required this.modelCardKeyFor,
    required this.isSameModel,
    required this.onNavigateToViewer,
    required this.onShowModelActions,
    required this.onAddNewTask,
  });

  @override
  State<TimePeelingSlot> createState() => TimePeelingSlotState();
}

class TimePeelingSlotState extends State<TimePeelingSlot> {
  late PageController _pageController;
  late final ValueNotifier<double> _pagePosition;

  /// Total items: create button first, then models (newest->oldest).
  int get _totalCount => widget.models.length + 1;

  @override
  void initState() {
    super.initState();
    _pageController = PageController(
      viewportFraction: kViewportFraction,
      initialPage: 1, // index 0 is create card, 1 is newest model
    );
    _pagePosition = ValueNotifier<double>(1.0);
    _pageController.addListener(_onScroll);
  }

  @override
  void dispose() {
    _pageController.removeListener(_onScroll);
    _pageController.dispose();
    _pagePosition.dispose();
    super.dispose();
  }

  void _onScroll() {
    final page = _pageController.page;
    if (page != null && (page - _pagePosition.value).abs() > 0.001) {
      _pagePosition.value = page;
    }
  }

  String _timeLabelFor(int modelIndex) {
    if (modelIndex < 0 || modelIndex >= widget.models.length) return '';
    final dt = DateTime.tryParse(
      widget.models[modelIndex]['created_at']?.toString() ?? '',
    );
    if (dt == null) return '--';
    final local = dt.toLocal();
    return '${local.month.toString().padLeft(2, '0')}/${local.day.toString().padLeft(2, '0')} '
        '${local.hour.toString().padLeft(2, '0')}:${local.minute.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    final slotBg = widget.isDark
        ? Colors.white.withValues(alpha: 0.04)
        : Colors.white.withValues(alpha: 0.55);
    final slotBorder = widget.isDark
        ? Colors.white.withValues(alpha: 0.07)
        : Colors.black.withValues(alpha: 0.06);
    final timeLabels = List<String>.generate(
      widget.models.length,
      _timeLabelFor,
      growable: false,
    );

    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 0, 12, 16),
      child: Container(
        decoration: BoxDecoration(
          color: slotBg,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: slotBorder, width: 1),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(
                alpha: widget.isDark ? 0.12 : 0.04,
              ),
              blurRadius: 12,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Title row
            Padding(
              padding: const EdgeInsets.fromLTRB(18, 14, 18, 0),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      widget.name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: widget.textColor,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 3,
                    ),
                    decoration: BoxDecoration(
                      color: widget.hintColor.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      '${widget.models.length}',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: widget.hintColor,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 10),
            SizedBox(
              height: kCarouselHeight,
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final fadeStop =
                      (kEdgeFadeWidth / constraints.maxWidth).clamp(0.0, 0.5);
                  return ShaderMask(
                    blendMode: BlendMode.dstIn,
                    shaderCallback: (rect) => LinearGradient(
                      begin: Alignment.centerLeft,
                      end: Alignment.centerRight,
                      colors: const [
                        Colors.transparent,
                        Colors.black,
                        Colors.black,
                        Colors.transparent,
                      ],
                      stops: [0.0, fadeStop, 1.0 - fadeStop, 1.0],
                    ).createShader(rect),
                    child: RepaintBoundary(
                      child: PageView.builder(
                        controller: _pageController,
                        itemCount: _totalCount,
                        clipBehavior: Clip.hardEdge,
                        itemBuilder: (context, index) {
                          if (index == 0) {
                            return _TimePeelingCarouselItem(
                              pagePosition: _pagePosition,
                              index: index,
                              child: _buildCreateCard(),
                            );
                          }

                          final model = widget.models[index - 1];
                          final cardKey = widget.modelCardKeyFor(model);
                          final isActionTarget = widget.isSameModel(
                            widget.activeModelAction,
                            model,
                          );

                          return _TimePeelingCarouselItem(
                            pagePosition: _pagePosition,
                            index: index,
                            forceHidden: isActionTarget,
                            child: IgnorePointer(
                              ignoring: isActionTarget,
                              child: GestureDetector(
                                onTap: () =>
                                    widget.onNavigateToViewer(model, null),
                                onLongPressStart: (_) => widget
                                    .onShowModelActions(model, imageOnly: true),
                                child: Container(
                                  key: cardKey,
                                  decoration: BoxDecoration(
                                    borderRadius: BorderRadius.circular(28),
                                    border: Border.all(
                                      color: widget.isDark
                                          ? Colors.white.withValues(alpha: 0.08)
                                          : Colors.black.withValues(alpha: 0.06),
                                      width: 1,
                                    ),
                                  ),
                                  child: RepaintBoundary(
                                    child: RecallModelTile(
                                      model: model,
                                      theme: widget.theme,
                                      isDark: widget.isDark,
                                      darkCard: widget.darkCard,
                                      darkInput: widget.darkInput,
                                      textColor: widget.textColor,
                                      hintTextColor: widget.hintTextColor,
                                      imageOnly: true,
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  );
                },
              ),
            ),
            SizedBox(
              height: 48,
              child: ValueListenableBuilder<double>(
                valueListenable: _pagePosition,
                builder: (context, currentPage, _) {
                  final selectedPage = currentPage.round().clamp(
                    0,
                    _totalCount - 1,
                  );
                  final timeLabel = selectedPage >= 1
                      ? timeLabels[selectedPage - 1]
                      : '';
                  return ClipRect(
                    child: CustomPaint(
                      painter: TimelinePainter(
                        modelCount: widget.models.length,
                        currentPage: currentPage,
                        timeLabel: timeLabel,
                        color: kTimelineColor,
                        viewportFraction: kViewportFraction,
                      ),
                      size: Size.infinite,
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCreateCard() {
    return GestureDetector(
      onTap: () => widget.onAddNewTask(widget.name),
      child: Container(
        decoration: BoxDecoration(
          color: widget.isDark
              ? Colors.white.withValues(alpha: 0.06)
              : Colors.white.withValues(alpha: 0.7),
          borderRadius: BorderRadius.circular(28),
          border: Border.all(
            color: kTimelineColor.withValues(alpha: 0.35),
            width: 1.5,
          ),
        ),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.add_rounded,
                size: 36,
                color: kTimelineColor.withValues(alpha: 0.7),
              ),
              const SizedBox(height: 6),
              Text(
                textLocalize("create"),
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: kTimelineColor.withValues(alpha: 0.7),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _TimePeelingCarouselItem extends StatelessWidget {
  final ValueNotifier<double> pagePosition;
  final int index;
  final Widget child;
  final bool forceHidden;

  const _TimePeelingCarouselItem({
    required this.pagePosition,
    required this.index,
    required this.child,
    this.forceHidden = false,
  });

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<double>(
      valueListenable: pagePosition,
      child: child,
      builder: (context, currentPage, child) {
        final distance = (index - currentPage).abs().clamp(0.0, 1.0);
        final scale = ui.lerpDouble(1.0, 0.82, distance)!;
        final opacity = forceHidden
            ? 0.0
            : ui.lerpDouble(1.0, 0.5, distance)!.clamp(0.0, 1.0);
        final shadowAlpha = ui.lerpDouble(0.18, 0.0, distance)!;

        return Center(
          child: Transform.scale(
            scale: scale,
            child: Opacity(
              opacity: opacity,
              child: DecoratedBox(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(30),
                  boxShadow: shadowAlpha <= 0.01
                      ? const []
                      : [
                          BoxShadow(
                            color: kTimelineColor.withValues(
                              alpha: shadowAlpha,
                            ),
                            blurRadius: ui.lerpDouble(18, 6, distance)!,
                            offset: const Offset(0, 6),
                          ),
                        ],
                ),
                child: child,
              ),
            ),
          ),
        );
      },
    );
  }
}

/// 时间线节点绘制器
///
/// 在轮播区域下方绘制连接线和选中节点的时间标签。
class TimelinePainter extends CustomPainter {
  /// Number of model cards (excludes the create card at index 0).
  final int modelCount;

  /// Current page position from PageController (index 0 = create card).
  final double currentPage;
  final String timeLabel;
  final Color color;
  final double viewportFraction;

  TimelinePainter({
    required this.modelCount,
    required this.currentPage,
    required this.color,
    required this.timeLabel,
    required this.viewportFraction,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (modelCount == 0) return;

    final lineY = 10.0;
    final slotWidth = size.width * viewportFraction;
    final centerX = size.width / 2;

    // Model cards occupy PageView indices 1..modelCount.
    // Node i (0-based model index) corresponds to PageView index (i + 1).
    // Its center X = centerX + (pageIndex - currentPage) * slotWidth.
    List<double> nodeXs = [];
    for (int i = 0; i < modelCount; i++) {
      final pageIndex = i + 1; // offset by create card
      final dx = centerX + (pageIndex - currentPage) * slotWidth;
      nodeXs.add(dx);
    }

    // Draw the connecting line between first and last node, clamped to bounds
    final linePaint = Paint()
      ..color = color.withValues(alpha: 0.25)
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final lineLeft = nodeXs.first.clamp(0.0, size.width);
    final lineRight = nodeXs.last.clamp(0.0, size.width);
    if (lineRight > lineLeft) {
      canvas.drawLine(
        Offset(lineLeft, lineY),
        Offset(lineRight, lineY),
        linePaint,
      );
    }

    // Selected model index (0-based): PageView selectedPage - 1
    final selectedPageIndex = currentPage.round().clamp(0, modelCount);
    // Convert to model index; -1 means create card is selected (no highlight)
    final selectedModelIndex = selectedPageIndex - 1;

    final normalRadius = 3.5;
    final selectedRadius = 5.5;

    for (int i = 0; i < modelCount; i++) {
      final x = nodeXs[i];
      if (x < -20 || x > size.width + 20) continue;

      // Distance from the selected model node (continuous for smooth animation)
      final distFromSelected = ((i + 1) - currentPage).abs().clamp(0.0, 1.0);
      final radius = ui.lerpDouble(
        selectedRadius,
        normalRadius,
        distFromSelected,
      )!;
      final alpha = ui.lerpDouble(0.95, 0.4, distFromSelected)!;

      final dotPaint = Paint()
        ..color = color.withValues(alpha: alpha)
        ..style = PaintingStyle.fill;

      canvas.drawCircle(Offset(x, lineY), radius, dotPaint);
    }

    // Draw time label below the selected model node
    if (timeLabel.isNotEmpty &&
        selectedModelIndex >= 0 &&
        selectedModelIndex < modelCount) {
      final labelX = nodeXs[selectedModelIndex];
      final textPainter = TextPainter(
        text: TextSpan(
          text: timeLabel,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: color.withValues(alpha: 0.85),
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final textX = (labelX - textPainter.width / 2).clamp(
        4.0,
        size.width - textPainter.width - 4,
      );
      textPainter.paint(canvas, Offset(textX, lineY + selectedRadius + 6));
    }
  }

  @override
  bool shouldRepaint(covariant TimelinePainter old) =>
      old.currentPage != currentPage ||
      old.modelCount != modelCount ||
      old.timeLabel != timeLabel;
}
