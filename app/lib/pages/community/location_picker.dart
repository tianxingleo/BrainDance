import 'dart:async';

import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../../services/location_service.dart';
import 'amap_search.dart';
import 'map_page.dart' show
    kCommunityMapTileUrl,
    kCommunityMapTileSubdomains,
    kCommunityMapTileUserAgent,
    kCommunityMapMinZoom,
    kCommunityMapMaxZoom;
import 'widgets/map_overlay_widgets.dart';
import 'widgets/map_search_widgets.dart';

/// LocationPickerPage 返回的结果。坐标始终是 GCJ-02。
class LocationPickResult {
  final String placeName;
  final double latitude;
  final double longitude;

  const LocationPickResult({
    required this.placeName,
    required this.latitude,
    required this.longitude,
  });
}

class LocationPickerPage extends StatefulWidget {
  /// 初始中心：传入用户先前的选点（GCJ-02），未选过则用 [fallback] 或杭州。
  final LatLng? initialCenter;
  final LatLng fallback;

  const LocationPickerPage({
    super.key,
    this.initialCenter,
    this.fallback = const LatLng(30.2741, 120.1551),
  });

  @override
  State<LocationPickerPage> createState() => _LocationPickerPageState();
}

class _LocationPickerPageState extends State<LocationPickerPage> {
  final MapController _controller = MapController();
  final TextEditingController _searchController = TextEditingController();
  final FocusNode _searchFocus = FocusNode();

  // 搜索状态
  String _searchKeyword = '';
  List<AmapPoi> _searchResults = const [];
  bool _searchLoading = false;
  String? _searchError;
  bool _searchResultsOpen = false;
  CancelToken? _searchCancelToken;
  int _searchSeq = 0;

  // 选点状态
  late LatLng _center;
  AmapRegeo? _regeo;
  bool _regeoLoading = false;
  String? _regeoError;
  CancelToken? _regeoCancelToken;
  Timer? _regeoDebounce;
  int _regeoSeq = 0;

  bool _locating = false;

  @override
  void initState() {
    super.initState();
    _center = widget.initialCenter ?? widget.fallback;
    // 进入后立刻反查一次中心点
    WidgetsBinding.instance.addPostFrameCallback((_) => _runRegeo(_center));
  }

  @override
  void dispose() {
    _regeoDebounce?.cancel();
    _regeoCancelToken?.cancel('disposed');
    _searchCancelToken?.cancel('disposed');
    _searchController.dispose();
    _searchFocus.dispose();
    _controller.dispose();
    super.dispose();
  }

  // ---------- 搜索 ----------

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
      _searchCancelToken = null;
    });
  }

  void _onSearchPoiTap(AmapPoi poi) {
    final currentZoom = _controller.camera.zoom;
    final targetZoom = currentZoom < 16 ? 16.0 : currentZoom;
    _controller.move(poi.location, targetZoom);
    setState(() {
      _center = poi.location;
      _regeo = AmapRegeo(
        placeName: poi.name,
        formattedAddress: poi.address,
        cityName: poi.cityName,
        district: poi.district,
      );
      _regeoLoading = false;
      _regeoError = null;
      _searchResultsOpen = false;
    });
    _searchFocus.unfocus();
    // 来自搜索结果的 placeName 已可信，跳过 regeo
  }

  // ---------- 拖动 → regeo ----------

  void _onMapEvent(MapEvent event) {
    if (event is MapEventMoveEnd ||
        event is MapEventFlingAnimationEnd ||
        event is MapEventScrollWheelZoom ||
        event is MapEventDoubleTapZoomEnd) {
      _scheduleRegeo();
    }
  }

  void _scheduleRegeo() {
    _regeoDebounce?.cancel();
    _regeoDebounce = Timer(const Duration(milliseconds: 250), () {
      if (!mounted) return;
      final c = _controller.camera.center;
      _runRegeo(c);
    });
  }

  Future<void> _runRegeo(LatLng point) async {
    _regeoCancelToken?.cancel('superseded');
    final token = CancelToken();
    final seq = ++_regeoSeq;
    setState(() {
      _center = point;
      _regeoLoading = true;
      _regeoError = null;
      _regeoCancelToken = token;
    });
    try {
      final r = await AmapSearchService.instance
          .regeoSearch(point, cancelToken: token);
      if (!mounted || seq != _regeoSeq) return;
      setState(() {
        _regeo = r;
        _regeoLoading = false;
      });
    } on AmapSearchException catch (e) {
      if (!mounted || seq != _regeoSeq) return;
      setState(() {
        _regeoLoading = false;
        _regeoError = e.message;
      });
    } on DioException catch (e) {
      if (CancelToken.isCancel(e)) return;
      if (!mounted || seq != _regeoSeq) return;
      setState(() {
        _regeoLoading = false;
        _regeoError = '网络异常，地址解析失败';
      });
    }
  }

  // ---------- 当前定位 ----------

  Future<void> _useCurrentLocation() async {
    if (_locating) return;
    setState(() => _locating = true);
    try {
      final p = await LocationService.instance.getCurrentGcj02();
      if (!mounted) return;
      _controller.move(p, 16);
      _runRegeo(p);
    } on LocationException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.message)),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('定位失败：$e')),
      );
    } finally {
      if (mounted) setState(() => _locating = false);
    }
  }

  // ---------- 完成 ----------

  void _confirm() {
    final r = _regeo;
    final name = (r?.placeName.isNotEmpty ?? false)
        ? r!.placeName
        : (r?.formattedAddress ?? '');
    Navigator.of(context).pop(LocationPickResult(
      placeName: name,
      latitude: _center.latitude,
      longitude: _center.longitude,
    ));
  }

  void _stepZoom(double delta) {
    final cam = _controller.camera;
    final next = (cam.zoom + delta)
        .clamp(kCommunityMapMinZoom, kCommunityMapMaxZoom)
        .toDouble();
    _controller.move(cam.center, next);
    _scheduleRegeo();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);
    final canConfirm = _regeo != null;

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: BDPageBackdrop(
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 20),
            child: Column(
              children: [
                _buildHeader(textColor, canConfirm),
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
                        (!_searchResultsOpen ||
                            _searchResults.isNotEmpty)) {
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
                              initialCenter: _center,
                              initialZoom: 15,
                              minZoom: kCommunityMapMinZoom,
                              maxZoom: kCommunityMapMaxZoom,
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
                                urlTemplate: kCommunityMapTileUrl,
                                subdomains: kCommunityMapTileSubdomains,
                                userAgentPackageName:
                                    kCommunityMapTileUserAgent,
                                maxZoom: kCommunityMapMaxZoom,
                                retinaMode:
                                    RetinaMode.isHighDensity(context),
                              ),
                            ],
                          ),
                          const MapCenterCrosshair(),
                          Positioned(
                            left: 14,
                            top: 14,
                            child: MapHintPill(
                              text: '拖动地图，对准目标点',
                              isDark: isDark,
                              hintColor: hintColor,
                            ),
                          ),
                          Positioned(
                            right: 12,
                            bottom: 12,
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              crossAxisAlignment: CrossAxisAlignment.end,
                              children: [
                                MapLocateButton(
                                  isDark: isDark,
                                  loading: _locating,
                                  onPressed: _useCurrentLocation,
                                ),
                                const SizedBox(height: 10),
                                MapZoomControls(
                                  isDark: isDark,
                                  onZoomIn: () => _stepZoom(1),
                                  onZoomOut: () => _stepZoom(-1),
                                ),
                              ],
                            ),
                          ),
                          if (_searchResultsOpen &&
                              _searchKeyword.isNotEmpty)
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
                                onTap: _onSearchPoiTap,
                                onClose: () => setState(
                                    () => _searchResultsOpen = false),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                _ConfirmCard(
                  center: _center,
                  regeo: _regeo,
                  loading: _regeoLoading,
                  error: _regeoError,
                  textColor: textColor,
                  hintColor: hintColor,
                  onConfirm: canConfirm ? _confirm : null,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(Color textColor, bool canConfirm) {
    return Row(
      children: [
        IconButton(
          onPressed: () => Navigator.of(context).maybePop(),
          icon: Icon(Icons.arrow_back_rounded, color: textColor),
        ),
        Expanded(
          child: Text(
            '选择位置',
            style: TextStyle(
              color: textColor,
              fontSize: 22,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.4,
            ),
          ),
        ),
        FilledButton(
          onPressed: canConfirm ? _confirm : null,
          child: const Text('完成'),
        ),
      ],
    );
  }
}

class _ConfirmCard extends StatelessWidget {
  final LatLng center;
  final AmapRegeo? regeo;
  final bool loading;
  final String? error;
  final Color textColor;
  final Color hintColor;
  final VoidCallback? onConfirm;

  const _ConfirmCard({
    required this.center,
    required this.regeo,
    required this.loading,
    required this.error,
    required this.textColor,
    required this.hintColor,
    required this.onConfirm,
  });

  @override
  Widget build(BuildContext context) {
    final placeName = regeo?.placeName.isNotEmpty == true
        ? regeo!.placeName
        : (regeo?.formattedAddress ?? (loading ? '正在解析地址…' : '未知位置'));
    final addressLine = regeo?.formattedAddress ?? '';
    final coordText =
        '${center.latitude.toStringAsFixed(6)}, ${center.longitude.toStringAsFixed(6)}';

    return BDPanelCard(
      padding: const EdgeInsets.fromLTRB(16, 14, 12, 14),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  color: BDDesign.colorMutedBlue.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(Icons.place_rounded,
                    color: BDDesign.colorMutedBlue, size: 20),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  placeName,
                  style: TextStyle(
                    color: textColor,
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if (loading)
                SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    valueColor:
                        AlwaysStoppedAnimation(BDDesign.colorMutedBlue),
                  ),
                ),
            ],
          ),
          if (addressLine.isNotEmpty) ...[
            const SizedBox(height: 6),
            Padding(
              padding: const EdgeInsets.only(left: 42),
              child: Text(
                addressLine,
                style: TextStyle(
                    color: hintColor, fontSize: 12.5, height: 1.4),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
          if (error != null) ...[
            const SizedBox(height: 6),
            Padding(
              padding: const EdgeInsets.only(left: 42),
              child: Text(
                error!,
                style: TextStyle(
                    color: Colors.redAccent.shade200,
                    fontSize: 12.5,
                    height: 1.4),
              ),
            ),
          ],
          const SizedBox(height: 10),
          Padding(
            padding: const EdgeInsets.only(left: 42),
            child: Row(
              children: [
                Icon(Icons.gps_fixed_rounded,
                    size: 13, color: hintColor.withValues(alpha: 0.85)),
                const SizedBox(width: 6),
                Text(
                  coordText,
                  style: TextStyle(
                    color: hintColor,
                    fontSize: 11.5,
                    fontFeatures: const [FontFeature.tabularFigures()],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              icon: const Icon(Icons.check_rounded, size: 18),
              label: const Text('选择此地点'),
              onPressed: onConfirm,
            ),
          ),
        ],
      ),
    );
  }
}