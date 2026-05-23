import 'dart:async';

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

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final textColor = isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = textColor.withValues(alpha: 0.55);
    final canConfirm = _regeo != null;

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      appBar: AppBar(
        title: const Text('选择位置'),
        actions: [
          TextButton(
            onPressed: canConfirm ? _confirm : null,
            child: const Text('完成'),
          ),
        ],
      ),
      body: SafeArea(
        top: false,
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
                  userAgentPackageName: kCommunityMapTileUserAgent,
                  maxZoom: kCommunityMapMaxZoom,
                  retinaMode: RetinaMode.isHighDensity(context),
                ),
              ],
            ),
            // 屏幕中心的十字准星
            IgnorePointer(
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.location_on_rounded,
                      color: BDDesign.colorMutedBlue,
                      size: 42,
                      shadows: const [
                        Shadow(
                          color: Color(0x66000000),
                          blurRadius: 6,
                          offset: Offset(0, 2),
                        ),
                      ],
                    ),
                    Container(
                      width: 8,
                      height: 8,
                      margin: const EdgeInsets.only(top: 36),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.4),
                        shape: BoxShape.circle,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            // 顶部搜索栏 + 浮层
            Positioned(
              top: 12,
              left: 12,
              right: 12,
              child: Column(
                children: [
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
                  if (_searchResultsOpen && _searchKeyword.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    SearchResultsOverlay(
                      isDark: isDark,
                      textColor: textColor,
                      hintColor: hintColor,
                      loading: _searchLoading,
                      error: _searchError,
                      results: _searchResults,
                      onTap: _onSearchPoiTap,
                      onClose: () =>
                          setState(() => _searchResultsOpen = false),
                    ),
                  ],
                ],
              ),
            ),
            // 右上角"使用当前位置"圆形按钮
            Positioned(
              right: 12,
              top: 76,
              child: _LocateFab(
                loading: _locating,
                onPressed: _useCurrentLocation,
              ),
            ),
            // 底部确认卡片
            Positioned(
              left: 12,
              right: 12,
              bottom: 12,
              child: _ConfirmCard(
                center: _center,
                regeo: _regeo,
                loading: _regeoLoading,
                error: _regeoError,
                textColor: textColor,
                hintColor: hintColor,
                onConfirm: canConfirm ? _confirm : null,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _LocateFab extends StatelessWidget {
  final bool loading;
  final VoidCallback onPressed;
  const _LocateFab({required this.loading, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Theme.of(context).colorScheme.surface,
      shape: const CircleBorder(),
      elevation: 3,
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: loading ? null : onPressed,
        child: SizedBox(
          width: 44,
          height: 44,
          child: loading
              ? const Center(
                  child: SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                )
              : const Icon(Icons.my_location_rounded, size: 22),
        ),
      ),
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
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.place_rounded,
                  color: BDDesign.colorMutedBlue, size: 20),
              const SizedBox(width: 8),
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
            const SizedBox(height: 4),
            Text(
              addressLine,
              style: TextStyle(
                  color: hintColor, fontSize: 12.5, height: 1.35),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ],
          if (error != null) ...[
            const SizedBox(height: 4),
            Text(
              error!,
              style: TextStyle(
                  color: Colors.redAccent.shade200,
                  fontSize: 12.5,
                  height: 1.35),
            ),
          ],
          const SizedBox(height: 6),
          Text(
            coordText,
            style: TextStyle(
              color: hintColor,
              fontSize: 11.5,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
          const SizedBox(height: 12),
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