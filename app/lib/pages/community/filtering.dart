import 'dart:math' as math;

import 'package:braindance/pages/community/models.dart';

/// Latitude/longitude bounds of the current map viewport.
class CommunityViewportBounds {
  final double north;
  final double south;
  final double east;
  final double west;

  const CommunityViewportBounds({
    required this.north,
    required this.south,
    required this.east,
    required this.west,
  });

  bool get isValid => north > south;

  /// East < west when the viewport crosses the antimeridian.
  bool get crossesAntimeridian => east < west;

  bool contains(double lat, double lng) {
    if (lat < south || lat > north) return false;
    if (crossesAntimeridian) {
      return lng >= west || lng <= east;
    }
    return lng >= west && lng <= east;
  }

  /// Approximate viewport center, used as the origin when applying a
  /// tag-radius filter without a sharper signal.
  ({double latitude, double longitude}) get center {
    final lat = (north + south) / 2;
    if (!crossesAntimeridian) {
      return (latitude: lat, longitude: (east + west) / 2);
    }
    // Wrap around: average across the antimeridian.
    final span = (180 - west) + (east + 180);
    final lng = ((west + span / 2) + 540) % 360 - 180;
    return (latitude: lat, longitude: lng);
  }
}

/// Zoom -> radius (km) lookup. Lower zoom = wider radius.
/// Discrete table keeps the first version stable; tweak before going smooth.
double tagRadiusKmForZoom(int zoom) {
  if (zoom <= 9) return 5.0;
  if (zoom <= 11) return 2.0;
  if (zoom <= 13) return 0.8;
  if (zoom <= 15) return 0.3;
  return 0.1;
}

/// Posts strictly inside the viewport bounds. When [bounds] is null we treat
/// the layer as "全量" — used as a fallback before a real bounds is known.
List<CommunityPost> filterPostsByBounds(
  List<CommunityPost> posts,
  CommunityViewportBounds? bounds,
) {
  if (bounds == null || !bounds.isValid) return posts;
  return posts
      .where((p) => bounds.contains(p.latitude, p.longitude))
      .toList(growable: false);
}

/// Posts that carry the given tag.
List<CommunityPost> filterPostsByTag(
  List<CommunityPost> posts,
  String? tag,
) {
  if (tag == null || tag.isEmpty) return posts;
  return posts.where((p) => p.tags.contains(tag)).toList(growable: false);
}

/// Posts within [radiusKm] of [origin]. Uses the equirectangular approximation
/// — accurate enough at neighborhood scale and cheap to compute.
List<CommunityPost> filterPostsByRadius(
  List<CommunityPost> posts,
  ({double latitude, double longitude}) origin,
  double radiusKm,
) {
  if (radiusKm <= 0) return posts;
  final radiusKmSq = radiusKm * radiusKm;
  return posts.where((p) {
    final km = _haversineKm(
      origin.latitude,
      origin.longitude,
      p.latitude,
      p.longitude,
    );
    return km * km <= radiusKmSq;
  }).toList(growable: false);
}

/// Tags that appear in the supplied posts, ranked by frequency. Used to drive
/// the explore-tab tag chips so they match the current viewport's content.
List<String> rankTagsFromPosts(
  List<CommunityPost> posts, {
  int limit = 12,
}) {
  if (posts.isEmpty) return const [];
  final counts = <String, int>{};
  for (final p in posts) {
    for (final raw in p.tags) {
      final tag = raw.trim();
      if (tag.isEmpty) continue;
      counts[tag] = (counts[tag] ?? 0) + 1;
    }
  }
  final entries = counts.entries.toList()
    ..sort((a, b) {
      final c = b.value.compareTo(a.value);
      if (c != 0) return c;
      return a.key.compareTo(b.key);
    });
  return entries.take(limit).map((e) => e.key).toList(growable: false);
}

double _haversineKm(double lat1, double lng1, double lat2, double lng2) {
  const earthKm = 6371.0;
  final dLat = _deg2rad(lat2 - lat1);
  final dLng = _deg2rad(lng2 - lng1);
  final a = math.sin(dLat / 2) * math.sin(dLat / 2) +
      math.cos(_deg2rad(lat1)) *
          math.cos(_deg2rad(lat2)) *
          math.sin(dLng / 2) *
          math.sin(dLng / 2);
  final c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a));
  return earthKm * c;
}

double _deg2rad(double deg) => deg * (math.pi / 180.0);
