import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';

/// Lightweight marker payload for the community map. Decoupled from the
/// full feed-style [CommunityPost] so the map data path stays cheap.
class CommunityMapMarker {
  final String id;
  final String title;
  final String placeName;
  final double latitude;
  final double longitude;
  final String? coverUrl;
  final DateTime createdAt;
  final int likeCount;
  final int viewCount;

  const CommunityMapMarker({
    required this.id,
    required this.title,
    required this.placeName,
    required this.latitude,
    required this.longitude,
    required this.createdAt,
    this.coverUrl,
    this.likeCount = 0,
    this.viewCount = 0,
  });

  LatLng get latLng => LatLng(latitude, longitude);

  bool get hasValidLocation =>
      latitude != 0 || longitude != 0;
}

/// Fixed geographic grid cell size (degrees) used for bucketing markers.
/// ~0.05° ≈ 5.5km at the equator; this is the unit of "stable distribution".
/// Bucket size is independent of the map zoom level on purpose.
const double _kBucketDegrees = 0.05;

String _bucketKey(double lat, double lng) {
  final yi = (lat / _kBucketDegrees).floor();
  final xi = (lng / _kBucketDegrees).floor();
  return '$yi:$xi';
}

/// "Fixed grid + density quota + stable order + hard cap" selection.
/// Output is deterministic for a given input and `limit`, so the marker
/// distribution does NOT change as the user zooms in/out.
List<CommunityMapMarker> selectMapMarkers(
  List<CommunityMapMarker> input, {
  required int limit,
}) {
  if (limit <= 0 || input.isEmpty) return const [];

  final valid = input.where((m) => m.hasValidLocation).toList();
  if (valid.isEmpty) return const [];
  if (valid.length <= limit) return _stableSort(valid);

  // 1) Bucket by fixed geographic grid.
  final Map<String, List<CommunityMapMarker>> buckets = {};
  for (final m in valid) {
    buckets.putIfAbsent(_bucketKey(m.latitude, m.longitude), () => []).add(m);
  }

  // 2) Sort each bucket by stable rank (newest first, then likes/views, id).
  for (final list in buckets.values) {
    _stableSortInPlace(list);
  }

  // 3) Density quota via Hamilton's method on bucket sizes.
  final bucketKeys = buckets.keys.toList()..sort();
  final sizes = [for (final k in bucketKeys) buckets[k]!.length];
  final totalPosts = sizes.fold<int>(0, (a, b) => a + b);

  final rawQuotas = <double>[
    for (final s in sizes) (s * limit) / totalPosts,
  ];
  final intQuotas = [for (final q in rawQuotas) q.floor()];
  // Cap each bucket by its actual content.
  for (var i = 0; i < intQuotas.length; i++) {
    if (intQuotas[i] > sizes[i]) intQuotas[i] = sizes[i];
  }
  var assigned = intQuotas.fold<int>(0, (a, b) => a + b);
  // Distribute remainders to the largest fractional parts (stable by key).
  if (assigned < limit) {
    final remainders = <MapEntry<int, double>>[
      for (var i = 0; i < rawQuotas.length; i++)
        MapEntry(i, rawQuotas[i] - intQuotas[i]),
    ]..sort((a, b) {
        final c = b.value.compareTo(a.value);
        if (c != 0) return c;
        return bucketKeys[a.key].compareTo(bucketKeys[b.key]);
      });
    var idx = 0;
    while (assigned < limit && idx < remainders.length) {
      final i = remainders[idx].key;
      if (intQuotas[i] < sizes[i]) {
        intQuotas[i]++;
        assigned++;
      }
      idx++;
      if (idx == remainders.length && assigned < limit) {
        // Re-loop for buckets that still have room.
        idx = 0;
        if (!intQuotas.asMap().entries.any((e) => e.value < sizes[e.key])) {
          break;
        }
      }
    }
  }

  // 4) Take quota from each bucket.
  final picked = <CommunityMapMarker>[];
  for (var i = 0; i < bucketKeys.length; i++) {
    final n = intQuotas[i];
    if (n <= 0) continue;
    picked.addAll(buckets[bucketKeys[i]]!.take(n));
  }

  // 5) Final defensive trim — quotas should already match `limit`.
  if (picked.length > limit) {
    _stableSortInPlace(picked);
    return picked.take(limit).toList();
  }
  return _stableSort(picked);
}

List<CommunityMapMarker> _stableSort(List<CommunityMapMarker> list) {
  final copy = [...list];
  _stableSortInPlace(copy);
  return copy;
}

void _stableSortInPlace(List<CommunityMapMarker> list) {
  list.sort((a, b) {
    final c = b.createdAt.compareTo(a.createdAt);
    if (c != 0) return c;
    final l = b.likeCount.compareTo(a.likeCount);
    if (l != 0) return l;
    final v = b.viewCount.compareTo(a.viewCount);
    if (v != 0) return v;
    return a.id.compareTo(b.id);
  });
}

// ---------------------------------------------------------------------------
// User preference: max marker count
// ---------------------------------------------------------------------------

class MarkerLimitPreference {
  static const _prefKey = 'community_map_marker_limit';
  static const presets = <int>[20, 50, 100, 200];
  static const defaultLimit = 50;
  static const minLimit = 5;
  static const maxLimit = 500;

  static Future<int> load() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final v = prefs.getInt(_prefKey);
      if (v == null) return defaultLimit;
      return v.clamp(minLimit, maxLimit);
    } catch (_) {
      return defaultLimit;
    }
  }

  static Future<void> save(int value) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt(_prefKey, value.clamp(minLimit, maxLimit));
    } catch (_) {}
  }
}

// ---------------------------------------------------------------------------
// Marker rendering layer
// ---------------------------------------------------------------------------

class CommunityMarkerLayer extends StatelessWidget {
  final List<CommunityMapMarker> markers;
  final void Function(CommunityMapMarker marker)? onTap;
  final void Function(CommunityMapMarker marker)? onLongPress;
  final bool interactive;

  const CommunityMarkerLayer({
    super.key,
    required this.markers,
    this.onTap,
    this.onLongPress,
    this.interactive = true,
  });

  @override
  Widget build(BuildContext context) {
    if (markers.isEmpty) return const SizedBox.shrink();
    final isDark = context.isDarkMode;
    return MarkerLayer(
      markers: [
        for (final m in markers)
          Marker(
            point: m.latLng,
            width: 36,
            height: 36,
            alignment: Alignment.topCenter,
            child: _MarkerDot(
              isDark: isDark,
              onTap: interactive && onTap != null ? () => onTap!(m) : null,
              onLongPress: interactive && onLongPress != null
                  ? () => onLongPress!(m)
                  : null,
            ),
          ),
      ],
    );
  }
}

class _MarkerDot extends StatelessWidget {
  final bool isDark;
  final VoidCallback? onTap;
  final VoidCallback? onLongPress;

  const _MarkerDot({
    required this.isDark,
    this.onTap,
    this.onLongPress,
  });

  @override
  Widget build(BuildContext context) {
    final core = isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final accent = BDDesign.colorMutedBlue;
    final shadow = (isDark ? Colors.black : Colors.black54).withValues(
      alpha: 0.35,
    );
    final pin = SizedBox(
      width: 22,
      height: 28,
      child: CustomPaint(
        painter: _PinPainter(
          fill: accent,
          stroke: core,
          shadow: shadow,
        ),
      ),
    );
    if (onTap == null && onLongPress == null) return Center(child: pin);
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: onTap,
      onLongPress: onLongPress,
      child: Center(child: pin),
    );
  }
}

class _PinPainter extends CustomPainter {
  final Color fill;
  final Color stroke;
  final Color shadow;

  _PinPainter({
    required this.fill,
    required this.stroke,
    required this.shadow,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;
    final cx = w / 2;
    final headR = w * 0.42;
    final headCenter = Offset(cx, headR + 1);

    final path = ui.Path()
      ..moveTo(cx, h)
      ..quadraticBezierTo(cx + headR * 1.05, h * 0.55, cx + headR, headR + 1)
      ..arcToPoint(
        Offset(cx - headR, headR + 1),
        radius: Radius.circular(headR),
        clockwise: false,
      )
      ..quadraticBezierTo(cx - headR * 1.05, h * 0.55, cx, h)
      ..close();

    canvas.drawPath(
      path.shift(const Offset(0, 1)),
      Paint()
        ..color = shadow
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 1.5),
    );
    canvas.drawPath(path, Paint()..color = fill);
    canvas.drawPath(
      path,
      Paint()
        ..color = stroke
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.2,
    );
    canvas.drawCircle(
      headCenter,
      headR * 0.42,
      Paint()..color = stroke,
    );
  }

  @override
  bool shouldRepaint(covariant _PinPainter old) =>
      old.fill != fill || old.stroke != stroke || old.shadow != shadow;
}
