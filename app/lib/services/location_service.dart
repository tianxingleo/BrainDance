import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';

import '../utils/geo_coord.dart';

enum LocationFailureReason {
  serviceDisabled,
  permissionDenied,
  permissionDeniedForever,
  timeout,
  unknown,
}

class LocationException implements Exception {
  final LocationFailureReason reason;
  final String message;
  const LocationException(this.reason, this.message);

  @override
  String toString() => 'LocationException($reason): $message';
}

/// 一次性获取当前位置，自动完成权限申请并把 WGS-84 → GCJ-02。
class LocationService {
  LocationService._();
  static final LocationService instance = LocationService._();

  /// 失败时抛 [LocationException]。
  Future<LatLng> getCurrentGcj02({
    Duration timeout = const Duration(seconds: 10),
  }) async {
    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      throw const LocationException(
        LocationFailureReason.serviceDisabled,
        '系统定位服务未开启，请在设置中打开。',
      );
    }

    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.deniedForever) {
      throw const LocationException(
        LocationFailureReason.permissionDeniedForever,
        '定位权限被永久拒绝，请到系统设置中允许。',
      );
    }
    if (permission == LocationPermission.denied) {
      throw const LocationException(
        LocationFailureReason.permissionDenied,
        '未授权定位权限。',
      );
    }

    try {
      final pos = await Geolocator.getCurrentPosition(
        locationSettings: LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: timeout,
        ),
      );
      return wgs84ToGcj02(LatLng(pos.latitude, pos.longitude));
    } catch (e) {
      throw LocationException(
        LocationFailureReason.timeout,
        '获取定位失败：$e',
      );
    }
  }
}
