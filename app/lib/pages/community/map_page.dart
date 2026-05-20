import 'dart:math' as math;

import 'package:braindance/configs/amap_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';

class CommunityMapViewport {
  final double latitude;
  final double longitude;
  final int zoom;

  const CommunityMapViewport({
    required this.latitude,
    required this.longitude,
    required this.zoom,
  });

  CommunityMapViewport copyWith({
    double? latitude,
    double? longitude,
    int? zoom,
  }) {
    return CommunityMapViewport(
      latitude: latitude ?? this.latitude,
      longitude: longitude ?? this.longitude,
      zoom: zoom ?? this.zoom,
    );
  }
}

class CommunityMapPage extends StatefulWidget {
  final CommunityMapViewport initialViewport;

  const CommunityMapPage({
    super.key,
    required this.initialViewport,
  });

  @override
  State<CommunityMapPage> createState() => _CommunityMapPageState();
}

class _CommunityMapPageState extends State<CommunityMapPage> {
  static const int _tileSize = 512;
  static const int _tileRadius = 1;
  static const int _maxDragTileExpansion = 3;
  static const List<int> _zoomPrefetchDeltas = <int>[-1, 0, 1];

  late CommunityMapViewport _viewport;
  Offset _dragOffset = Offset.zero;
  int _gestureStartZoom = 10;
  bool _returningViewport = false;

  @override
  void initState() {
    super.initState();
    _viewport = _normalized(widget.initialViewport);
  }

  CommunityMapViewport _normalized(CommunityMapViewport viewport) {
    return CommunityMapViewport(
      latitude: viewport.latitude.clamp(-85.0, 85.0).toDouble(),
      longitude: _wrapLongitude(viewport.longitude),
      zoom: viewport.zoom.clamp(1, 17).toInt(),
    );
  }

  double _wrapLongitude(double longitude) {
    var value = longitude;
    while (value > 180) {
      value -= 360;
    }
    while (value < -180) {
      value += 360;
    }
    return value;
  }

  double _degreesPerPixelFor(int zoom) => 360 / (_tileSize * math.pow(2, zoom));

  void _finish() {
    _returningViewport = true;
    Navigator.pop(context, _viewport);
  }

  void _onScaleStart(ScaleStartDetails details) {
    _gestureStartZoom = _viewport.zoom;
  }

  void _onScaleUpdate(ScaleUpdateDetails details) {
    if (details.pointerCount >= 2) {
      _commitZoomForScale(details.scale);
      return;
    }

    setState(() {
      _dragOffset += details.focalPointDelta;
    });
  }

  void _onScaleEnd(ScaleEndDetails details) {
    _commitDragOffset();
  }

  void _commitDragOffset() {
    if (_dragOffset == Offset.zero) return;
    setState(() {
      _viewport = _shiftViewportByPixels(_viewport, -_dragOffset);
      _dragOffset = Offset.zero;
    });
  }

  void _commitZoomForScale(double scale) {
    final zoomDelta = (math.log(scale) / math.log(1.45)).round();
    final nextZoom = (_gestureStartZoom + zoomDelta).clamp(1, 17).toInt();
    if (nextZoom == _viewport.zoom) return;
    setState(() {
      _viewport = _viewport.copyWith(zoom: nextZoom);
      _dragOffset = Offset.zero;
    });
  }

  CommunityMapViewport _shiftViewportByPixels(
    CommunityMapViewport base,
    Offset pixelDelta,
  ) {
    final latScale = _safeCos(base.latitude);
    final degreesPerPixel = _degreesPerPixelFor(base.zoom);
    final lngDelta = pixelDelta.dx * degreesPerPixel / latScale;
    final latDelta = -pixelDelta.dy * degreesPerPixel;
    return _normalized(
      base.copyWith(
        latitude: base.latitude + latDelta,
        longitude: base.longitude + lngDelta,
      ),
    );
  }

  double _safeCos(double latitude) {
    return math.max(0.18, math.cos(latitude * math.pi / 180).abs());
  }

  List<_AmapTileSpec> _buildTiles() {
    final tiles = <_AmapTileSpec>[];
    final xRange = _tileRangeForAxis(_dragOffset.dx);
    final yRange = _tileRangeForAxis(_dragOffset.dy);
    final zoomLevels = _zoomPrefetchDeltas
        .map((delta) => (_viewport.zoom + delta).clamp(1, 17).toInt())
        .toSet()
        .toList();
    for (final zoom in zoomLevels) {
      final zoomViewport = _viewport.copyWith(zoom: zoom);
      final visible = zoom == _viewport.zoom;
      for (var y = yRange.start; y <= yRange.end; y++) {
        for (var x = xRange.start; x <= xRange.end; x++) {
          final tileCenter = _shiftViewportByPixels(
            zoomViewport,
            Offset(x * _tileSize.toDouble(), y * _tileSize.toDouble()),
          );
          tiles.add(
            _AmapTileSpec(
              key: ValueKey(
                '$zoom:$x:$y',
              ),
              viewport: tileCenter,
              offset: Offset(
                x * _tileSize.toDouble(),
                y * _tileSize.toDouble(),
              ),
              visible: visible,
            ),
          );
        }
      }
    }
    return tiles;
  }

  _TileIndexRange _tileRangeForAxis(double dragDelta) {
    final baseStart = -_tileRadius;
    final baseEnd = _tileRadius;
    final extraNegative = dragDelta > 0
        ? (dragDelta / _tileSize).ceil().clamp(0, _maxDragTileExpansion)
        : 0;
    final extraPositive = dragDelta < 0
        ? (-dragDelta / _tileSize).ceil().clamp(0, _maxDragTileExpansion)
        : 0;
    return _TileIndexRange(
      baseStart - extraNegative,
      baseEnd + extraPositive,
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) {
        if (didPop || _returningViewport) return;
        _finish();
      },
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: BDPageBackdrop(
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 20),
              child: Column(
                children: [
                  Row(
                    children: [
                      IconButton(
                        onPressed: _finish,
                        icon: Icon(Icons.arrow_back_rounded, color: textColor),
                      ),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '高德静态地图',
                              style: TextStyle(
                                color: textColor,
                                fontSize: 20,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              '${_viewport.latitude.toStringAsFixed(5)}, ${_viewport.longitude.toStringAsFixed(5)} · Zoom ${_viewport.zoom}',
                              style:
                                  TextStyle(color: hintColor, fontSize: 12.5),
                            ),
                          ],
                        ),
                      ),
                      FilledButton(
                        onPressed: _finish,
                        child: const Text('完成'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Expanded(
                    child: GestureDetector(
                      onScaleStart: _onScaleStart,
                      onScaleUpdate: _onScaleUpdate,
                      onScaleEnd: _onScaleEnd,
                      child: BDPanelCard(
                        padding: EdgeInsets.zero,
                        child: ClipRRect(
                          borderRadius: BDDesign.radiusLarge,
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              _AmapTileGrid(
                                tiles: _buildTiles(),
                                dragOffset: _dragOffset,
                              ),
                              _MapHintPill(
                                text: '单指拖拽，双指缩放',
                                isDark: isDark,
                                hintColor: hintColor,
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 14),
                  BDPanelCard(
                    padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
                    child: Row(
                      children: [
                        Icon(Icons.travel_explore_rounded, color: hintColor),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            '退出后，社区页地图组件会显示当前中心点和缩放级别。',
                            style: TextStyle(color: hintColor, height: 1.35),
                          ),
                        ),
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
}

class CommunityAmapPreview extends StatelessWidget {
  final CommunityMapViewport viewport;
  final int width;
  final int height;

  const CommunityAmapPreview({
    super.key,
    required this.viewport,
    this.width = 640,
    this.height = 372,
  });

  @override
  Widget build(BuildContext context) {
    return _AmapStaticMapImage(
      viewport: viewport,
      width: width,
      height: height,
      quietErrors: false,
    );
  }
}

class _AmapTileGrid extends StatelessWidget {
  final List<_AmapTileSpec> tiles;
  final Offset dragOffset;

  const _AmapTileGrid({
    required this.tiles,
    required this.dragOffset,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final center = Offset(
          constraints.maxWidth / 2,
          constraints.maxHeight / 2,
        );
        return Stack(
          clipBehavior: Clip.hardEdge,
          children: [
            for (final tile in tiles)
              Positioned(
                key: tile.key,
                left: center.dx - _CommunityMapPageState._tileSize / 2 +
                    tile.offset.dx +
                    dragOffset.dx,
                top: center.dy - _CommunityMapPageState._tileSize / 2 +
                    tile.offset.dy +
                    dragOffset.dy,
                width: _CommunityMapPageState._tileSize.toDouble(),
                height: _CommunityMapPageState._tileSize.toDouble(),
                child: Offstage(
                  offstage: !tile.visible,
                  child: _AmapStaticMapImage(
                    viewport: tile.viewport,
                    width: _CommunityMapPageState._tileSize,
                    height: _CommunityMapPageState._tileSize,
                    quietErrors: true,
                  ),
                ),
              ),
          ],
        );
      },
    );
  }
}

class _AmapTileSpec {
  final Key key;
  final CommunityMapViewport viewport;
  final Offset offset;
  final bool visible;

  const _AmapTileSpec({
    required this.key,
    required this.viewport,
    required this.offset,
    required this.visible,
  });
}

class _TileIndexRange {
  final int start;
  final int end;

  const _TileIndexRange(this.start, this.end);
}

class _AmapStaticMapImage extends StatelessWidget {
  final CommunityMapViewport viewport;
  final int width;
  final int height;
  final bool quietErrors;

  const _AmapStaticMapImage({
    required this.viewport,
    required this.width,
    required this.height,
    this.quietErrors = false,
  });

  @override
  Widget build(BuildContext context) {
    if (!AmapConfig.hasWebServiceKey) {
      return const _MapPlaceholder(text: '缺少 AMAP_WEB_SERVICE_KEY');
    }

    final uri = AmapConfig.staticMapUri(
      latitude: viewport.latitude,
      longitude: viewport.longitude,
      zoom: viewport.zoom,
      width: width,
      height: height,
    );

    return Image.network(
      uri.toString(),
      fit: BoxFit.cover,
      filterQuality: FilterQuality.low,
      gaplessPlayback: true,
      loadingBuilder: (context, child, progress) {
        if (progress == null) return child;
        if (quietErrors) {
          return child;
        }
        final total = progress.expectedTotalBytes;
        return _MapPlaceholder(
          text: '地图加载中',
          progress:
              total == null ? null : progress.cumulativeBytesLoaded / total,
        );
      },
      errorBuilder: (context, error, stackTrace) {
        if (quietErrors) {
          return _TileSoftPlaceholder(isDark: context.isDarkMode);
        }
        return const _MapPlaceholder(text: '高德静态地图暂不可用');
      },
    );
  }
}

class _TileSoftPlaceholder extends StatelessWidget {
  final bool isDark;

  const _TileSoftPlaceholder({required this.isDark});

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: isDark ? AppTheme.darkSurfaceElevated : const Color(0xFFEAF1F8),
    );
  }
}

class _MapHintPill extends StatelessWidget {
  final String text;
  final bool isDark;
  final Color hintColor;

  const _MapHintPill({
    required this.text,
    required this.isDark,
    required this.hintColor,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: 14,
      top: 14,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: (isDark ? Colors.black : Colors.white).withValues(alpha: 0.76),
          borderRadius: BorderRadius.circular(999),
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          child: Text(
            text,
            style: TextStyle(color: hintColor, fontSize: 12),
          ),
        ),
      ),
    );
  }
}

class _MapPlaceholder extends StatelessWidget {
  final String text;
  final double? progress;

  const _MapPlaceholder({
    required this.text,
    this.progress,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return ColoredBox(
      color: isDark ? AppTheme.darkSurfaceElevated : const Color(0xFFEAF1F8),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.map_outlined, size: 40, color: hintColor),
            const SizedBox(height: 10),
            Text(
              text,
              style: TextStyle(color: textColor, fontWeight: FontWeight.w700),
            ),
            if (progress != null) ...[
              const SizedBox(height: 12),
              SizedBox(
                width: 120,
                child: LinearProgressIndicator(value: progress),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
