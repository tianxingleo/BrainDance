import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import 'amap_search.dart';
import 'filtering.dart';
import 'map_marker.dart';
import 'repository.dart';
import 'widgets/map_search_widgets.dart';

const String kCommunityMapTileUrl =
    'https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}';
const List<String> kCommunityMapTileSubdomains = ['1', '2', '3', '4'];
const String kCommunityMapTileUserAgent = 'com.braindance.app';
const double kCommunityMapMinZoom = 2;
const double kCommunityMapMaxZoom = 18;
const int _kPreviewMarkerLimit = 30;

const String _tileUrl = kCommunityMapTileUrl;
const List<String> _tileSubdomains = kCommunityMapTileSubdomains;
const String _tileUserAgent = kCommunityMapTileUserAgent;
const double _kMinZoom = kCommunityMapMinZoom;
const double _kMaxZoom = kCommunityMapMaxZoom;

class CommunityMapViewport {
  final double latitude;
  final double longitude;
  final int zoom;
  final CommunityViewportBounds? bounds;

  const CommunityMapViewport({
    required this.latitude,
    required this.longitude,
    required this.zoom,
    this.bounds,
  });

  CommunityMapViewport copyWith({
    double? latitude,
    double? longitude,
    int? zoom,
    CommunityViewportBounds? bounds,
  }) {
    return CommunityMapViewport(
      latitude: latitude ?? this.latitude,
      longitude: longitude ?? this.longitude,
      zoom: zoom ?? this.zoom,
      bounds: bounds ?? this.bounds,
    );
  }

  LatLng get latLng => LatLng(latitude, longitude);
}

CommunityMapViewport _viewportFromCamera(MapCamera camera) {
  final visible = camera.visibleBounds;
  return CommunityMapViewport(
    latitude: camera.center.latitude,
    longitude: camera.center.longitude,
    zoom: camera.zoom.round().clamp(_kMinZoom.toInt(), _kMaxZoom.toInt()),
    bounds: CommunityViewportBounds(
      north: visible.north,
      south: visible.south,
      east: visible.east,
      west: visible.west,
    ),
  );
}

bool _isCameraFinite(MapCamera camera) {
  final c = camera.center;
  if (!c.latitude.isFinite || !c.longitude.isFinite) return false;
  if (!camera.zoom.isFinite) return false;
  final b = camera.visibleBounds;
  return b.north.isFinite &&
      b.south.isFinite &&
      b.east.isFinite &&
      b.west.isFinite;
}

class CommunityMapPage extends StatefulWidget {
  final CommunityMapViewport initialViewport;

  /// Tap a marker on the map. The map page already exposes the marker payload
  /// (id, title, location) — host pages decide whether to open detail.
  final void Function(CommunityMapMarker marker)? onMarkerTap;

  /// Long-press a marker — used by the host to open the location aggregation
  /// sheet ("posts at this place"). Falls back to [onMarkerTap] when null.
  final void Function(CommunityMapMarker marker)? onMarkerLongPress;

  const CommunityMapPage({
    super.key,
    required this.initialViewport,
    this.onMarkerTap,
    this.onMarkerLongPress,
  });

  @override
  State<CommunityMapPage> createState() => _CommunityMapPageState();
}

class _CommunityMapPageState extends State<CommunityMapPage> {
  final MapController _controller = MapController();
  final CommunityRepository _repository = CommunityRepository();
  final TextEditingController _searchController = TextEditingController();
  final FocusNode _searchFocus = FocusNode();
  late CommunityMapViewport _viewport;
  bool _returningViewport = false;
  List<CommunityMapMarker> _allMarkers = const [];
  List<CommunityMapMarker> _visibleMarkers = const [];
  int _markerLimit = MarkerLimitPreference.defaultLimit;
  bool _markersLoading = true;

  // 搜索状态
  String _searchKeyword = '';
  List<AmapPoi> _searchResults = const [];
  bool _searchLoading = false;
  String? _searchError;
  AmapPoi? _selectedSearchPoi;
  bool _searchResultsOpen = false;
  CancelToken? _searchCancelToken;
  int _searchSeq = 0;

  @override
  void initState() {
    super.initState();
    _viewport = widget.initialViewport;
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    final limit = await MarkerLimitPreference.load();
    if (!mounted) return;
    setState(() => _markerLimit = limit);
    await _loadMarkers();
  }

  Future<void> _loadMarkers() async {
    if (!mounted) return;
    setState(() => _markersLoading = true);
    final markers = await _repository.fetchMapMarkers();
    if (!mounted) return;
    setState(() {
      _allMarkers = markers;
      _visibleMarkers = selectMapMarkers(markers, limit: _markerLimit);
      _markersLoading = false;
    });
  }

  void _applyMarkerLimit(int next) {
    final clamped = next.clamp(
      MarkerLimitPreference.minLimit,
      MarkerLimitPreference.maxLimit,
    );
    if (clamped == _markerLimit) return;
    setState(() {
      _markerLimit = clamped;
      _visibleMarkers = selectMapMarkers(_allMarkers, limit: clamped);
    });
    MarkerLimitPreference.save(clamped);
  }

  @override
  void dispose() {
    _searchCancelToken?.cancel('disposed');
    _searchController.dispose();
    _searchFocus.dispose();
    _controller.dispose();
    super.dispose();
  }

  void _finish() {
    _returningViewport = true;
    Navigator.pop(context, _viewport);
  }

  void _onMapEvent(MapEvent event) {
    if (!_isCameraFinite(event.camera)) return;
    final updated = _viewportFromCamera(event.camera);
    final prev = _viewport;
    final boundsChanged = prev.bounds == null ||
        updated.bounds == null ||
        prev.bounds!.north != updated.bounds!.north ||
        prev.bounds!.south != updated.bounds!.south ||
        prev.bounds!.east != updated.bounds!.east ||
        prev.bounds!.west != updated.bounds!.west;
    if (!boundsChanged &&
        updated.latitude == prev.latitude &&
        updated.longitude == prev.longitude &&
        updated.zoom == prev.zoom) {
      return;
    }
    setState(() => _viewport = updated);
  }

  void _stepZoom(int delta) {
    final camera = _controller.camera;
    if (!_isCameraFinite(camera)) return;
    final next =
        (camera.zoom + delta).clamp(_kMinZoom, _kMaxZoom).toDouble();
    _controller.move(camera.center, next);
  }

  void _handleMarkerTap(CommunityMapMarker marker) {
    if (widget.onMarkerTap != null) {
      widget.onMarkerTap!(marker);
      return;
    }
    _controller.move(marker.latLng, _controller.camera.zoom);
  }

  void _handleMarkerLongPress(CommunityMapMarker marker) {
    final cb = widget.onMarkerLongPress ?? widget.onMarkerTap;
    if (cb != null) cb(marker);
  }

  Future<void> _openLimitSheet() async {
    final picked = await showModalBottomSheet<int>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (ctx) => _MarkerLimitSheet(current: _markerLimit),
    );
    if (picked != null) _applyMarkerLimit(picked);
  }

  Future<void> _runSearch(String raw) async {
    final keyword = raw.trim();
    if (keyword.isEmpty) {
      _clearSearch();
      return;
    }
    _searchCancelToken?.cancel('superseded');
    final token = CancelToken();
    final seq = ++_searchSeq;
    setState(() {
      _searchKeyword = keyword;
      _searchLoading = true;
      _searchError = null;
      _searchResultsOpen = true;
      _searchCancelToken = token;
    });
    try {
      final results = await AmapSearchService.instance.searchByText(
        keyword,
        cancelToken: token,
      );
      if (!mounted || seq != _searchSeq) return;
      setState(() {
        _searchResults = results;
        _searchLoading = false;
      });
    } on AmapSearchException catch (e) {
      if (!mounted || seq != _searchSeq) return;
      setState(() {
        _searchResults = const [];
        _searchLoading = false;
        _searchError = e.message;
      });
    } on DioException catch (e) {
      if (CancelToken.isCancel(e)) return;
      if (!mounted || seq != _searchSeq) return;
      setState(() {
        _searchResults = const [];
        _searchLoading = false;
        _searchError = '网络异常，请稍后重试';
      });
    }
  }

  void _clearSearch() {
    _searchCancelToken?.cancel('cleared');
    _searchController.clear();
    setState(() {
      _searchKeyword = '';
      _searchResults = const [];
      _searchLoading = false;
      _searchError = null;
      _searchResultsOpen = false;
      _selectedSearchPoi = null;
      _searchCancelToken = null;
    });
  }

  void _handlePoiTap(AmapPoi poi) {
    final currentZoom = _controller.camera.zoom;
    final targetZoom = currentZoom < 15 ? 15.0 : currentZoom;
    _controller.move(poi.location, targetZoom);
    setState(() {
      _selectedSearchPoi = poi;
      _searchResultsOpen = false;
    });
    _searchFocus.unfocus();
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
                              '社区地图',
                              style: TextStyle(
                                color: textColor,
                                fontSize: 20,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              _markersLoading
                                  ? '${_viewport.latitude.toStringAsFixed(5)}, ${_viewport.longitude.toStringAsFixed(5)} · Zoom ${_viewport.zoom}'
                                  : '${_visibleMarkers.length}/${_allMarkers.length} 个标记 · 上限 $_markerLimit',
                              style:
                                  TextStyle(color: hintColor, fontSize: 12.5),
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        tooltip: '标记上限',
                        onPressed: _openLimitSheet,
                        icon: Icon(Icons.tune_rounded, color: textColor),
                      ),
                      FilledButton(
                        onPressed: _finish,
                        child: const Text('完成'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  MapSearchBar(
                    controller: _searchController,
                    focusNode: _searchFocus,
                    isDark: isDark,
                    textColor: textColor,
                    hintColor: hintColor,
                    loading: _searchLoading,
                    hasKeyword: _searchKeyword.isNotEmpty,
                    onSubmitted: _runSearch,
                    onClear: _clearSearch,
                    onFocusResults: () {
                      if (_searchKeyword.isNotEmpty &&
                          (!_searchResultsOpen || _searchResults.isNotEmpty)) {
                        setState(() => _searchResultsOpen = true);
                      }
                    },
                  ),
                  const SizedBox(height: 14),
                  Expanded(
                    child: BDPanelCard(
                      padding: EdgeInsets.zero,
                      child: ClipRRect(
                        borderRadius: BDDesign.radiusLarge,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            FlutterMap(
                              mapController: _controller,
                              options: MapOptions(
                                initialCenter: _viewport.latLng,
                                initialZoom: _viewport.zoom.toDouble(),
                                minZoom: _kMinZoom,
                                maxZoom: _kMaxZoom,
                                interactionOptions: const InteractionOptions(
                                  flags: InteractiveFlag.drag |
                                      InteractiveFlag.flingAnimation |
                                      InteractiveFlag.pinchZoom |
                                      InteractiveFlag.doubleTapZoom |
                                      InteractiveFlag.scrollWheelZoom,
                                ),
                                onMapEvent: _onMapEvent,
                              ),
                              children: [
                                TileLayer(
                                  urlTemplate: _tileUrl,
                                  subdomains: _tileSubdomains,
                                  userAgentPackageName: _tileUserAgent,
                                  maxZoom: _kMaxZoom,
                                  retinaMode:
                                      RetinaMode.isHighDensity(context),
                                ),
                                CommunityMarkerLayer(
                                  markers: _visibleMarkers,
                                  onTap: _handleMarkerTap,
                                  onLongPress: _handleMarkerLongPress,
                                ),
                                if (_selectedSearchPoi != null)
                                  SearchPinLayer(poi: _selectedSearchPoi!),
                              ],
                            ),
                            _MapHintPill(
                              text: '单指拖拽，双指缩放',
                              isDark: isDark,
                              hintColor: hintColor,
                            ),
                            Positioned(
                              right: 12,
                              bottom: 12,
                              child: _ZoomControls(
                                isDark: isDark,
                                onZoomIn: () => _stepZoom(1),
                                onZoomOut: () => _stepZoom(-1),
                              ),
                            ),
                            if (_searchResultsOpen && _searchKeyword.isNotEmpty)
                              Positioned(
                                left: 8,
                                right: 8,
                                top: 8,
                                child: SearchResultsOverlay(
                                  isDark: isDark,
                                  textColor: textColor,
                                  hintColor: hintColor,
                                  loading: _searchLoading,
                                  error: _searchError,
                                  results: _searchResults,
                                  onTap: _handlePoiTap,
                                  onClose: () => setState(
                                      () => _searchResultsOpen = false),
                                ),
                              ),
                          ],
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
                            _markersLoading
                                ? '正在加载社区标记…'
                                : '点击标记进入帖子，长按查看同地点列表。缩放不会改变标记分布。',
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

/// Non-interactive preview used as a thumbnail on the community Explore tab.
/// Name kept for source compatibility with [CommunityExploreView].
class CommunityAmapPreview extends StatefulWidget {
  final CommunityMapViewport viewport;
  final int width;
  final int height;
  final List<CommunityMapMarker> markers;

  const CommunityAmapPreview({
    super.key,
    required this.viewport,
    this.width = 640,
    this.height = 372,
    this.markers = const [],
  });

  @override
  State<CommunityAmapPreview> createState() => _CommunityAmapPreviewState();
}

class _CommunityAmapPreviewState extends State<CommunityAmapPreview> {
  late final MapController _controller = MapController();
  late List<CommunityMapMarker> _previewMarkers;

  @override
  void initState() {
    super.initState();
    _previewMarkers = selectMapMarkers(
      widget.markers,
      limit: _kPreviewMarkerLimit,
    );
  }

  @override
  void didUpdateWidget(covariant CommunityAmapPreview oldWidget) {
    super.didUpdateWidget(oldWidget);
    final v = widget.viewport;
    final old = oldWidget.viewport;
    if (v.latitude != old.latitude ||
        v.longitude != old.longitude ||
        v.zoom != old.zoom) {
      _controller.move(v.latLng, v.zoom.toDouble());
    }
    if (!identical(widget.markers, oldWidget.markers)) {
      _previewMarkers = selectMapMarkers(
        widget.markers,
        limit: _kPreviewMarkerLimit,
      );
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: FlutterMap(
        mapController: _controller,
        options: MapOptions(
          initialCenter: widget.viewport.latLng,
          initialZoom: widget.viewport.zoom.toDouble(),
          minZoom: _kMinZoom,
          maxZoom: _kMaxZoom,
          interactionOptions: const InteractionOptions(
            flags: InteractiveFlag.none,
          ),
        ),
        children: [
          TileLayer(
            urlTemplate: _tileUrl,
            subdomains: _tileSubdomains,
            userAgentPackageName: _tileUserAgent,
            maxZoom: _kMaxZoom,
            retinaMode: RetinaMode.isHighDensity(context),
          ),
          CommunityMarkerLayer(
            markers: _previewMarkers,
            interactive: false,
          ),
        ],
      ),
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

class _ZoomControls extends StatelessWidget {
  final bool isDark;
  final VoidCallback onZoomIn;
  final VoidCallback onZoomOut;

  const _ZoomControls({
    required this.isDark,
    required this.onZoomIn,
    required this.onZoomOut,
  });

  @override
  Widget build(BuildContext context) {
    final bgColor =
        (isDark ? Colors.black : Colors.white).withValues(alpha: 0.82);
    final iconColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            onPressed: onZoomIn,
            icon: Icon(Icons.add_rounded, color: iconColor),
          ),
          Container(
            width: 24,
            height: 1,
            color: iconColor.withValues(alpha: 0.18),
          ),
          IconButton(
            onPressed: onZoomOut,
            icon: Icon(Icons.remove_rounded, color: iconColor),
          ),
        ],
      ),
    );
  }
}

class _MarkerLimitSheet extends StatefulWidget {
  final int current;

  const _MarkerLimitSheet({required this.current});

  @override
  State<_MarkerLimitSheet> createState() => _MarkerLimitSheetState();
}

class _MarkerLimitSheetState extends State<_MarkerLimitSheet> {
  late int _value;
  late TextEditingController _customController;

  @override
  void initState() {
    super.initState();
    _value = widget.current;
    _customController = TextEditingController(text: '$_value');
  }

  @override
  void dispose() {
    _customController.dispose();
    super.dispose();
  }

  void _select(int next) {
    setState(() {
      _value = next.clamp(
        MarkerLimitPreference.minLimit,
        MarkerLimitPreference.maxLimit,
      );
      _customController.text = '$_value';
    });
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);
    return Padding(
      padding: EdgeInsets.fromLTRB(
        16,
        20,
        16,
        20 + MediaQuery.viewInsetsOf(context).bottom,
      ),
      child: BDPanelCard(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 16),
        child: SafeArea(
          top: false,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '最大标记数',
                style: TextStyle(
                  color: textColor,
                  fontSize: 18,
                  fontWeight: FontWeight.w800,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                '决定地图上同时显示多少个帖子标记。算法会保证缩放时分布稳定。',
                style: TextStyle(color: hintColor, height: 1.4),
              ),
              const SizedBox(height: 14),
              Wrap(
                spacing: 10,
                runSpacing: 10,
                children: [
                  for (final preset in MarkerLimitPreference.presets)
                    ChoiceChip(
                      label: Text('$preset'),
                      selected: _value == preset,
                      onSelected: (_) => _select(preset),
                    ),
                ],
              ),
              const SizedBox(height: 14),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _customController,
                      keyboardType: TextInputType.number,
                      decoration: const InputDecoration(
                        labelText: '自定义',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                      onChanged: (raw) {
                        final parsed = int.tryParse(raw.trim());
                        if (parsed != null) {
                          setState(() {
                            _value = parsed.clamp(
                              MarkerLimitPreference.minLimit,
                              MarkerLimitPreference.maxLimit,
                            );
                          });
                        }
                      },
                    ),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    '范围 ${MarkerLimitPreference.minLimit}–${MarkerLimitPreference.maxLimit}',
                    style: TextStyle(color: hintColor, fontSize: 12.5),
                  ),
                ],
              ),
              const SizedBox(height: 18),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('取消'),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    onPressed: () => Navigator.pop(context, _value),
                    child: const Text('应用'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
