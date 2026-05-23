import 'dart:async';

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
  ///
  /// 策略：
  /// 1. 校验定位服务与权限。
  /// 2. 在 Dart 端用 `.timeout` 包裹 `getCurrentPosition`，避免 geolocator
  ///    内部 `timeLimit` 在冷启动 GPS 时直接抛 `TimeoutException: Future not
  ///    completed` 且不释放底层订阅的问题。
  /// 3. 超时或失败时回落到 `getLastKnownPosition`，仅在两者都拿不到时抛错。
  Future<LatLng> getCurrentGcj02({
    Duration timeout = const Duration(seconds: 15),
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

    Position? pos;
    Object? lastError;
    try {
      pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
        ),
      ).timeout(timeout);
    } on TimeoutException catch (e) {
      lastError = e;
    } catch (e) {
      lastError = e;
    }

    if (pos == null) {
      try {
        pos = await Geolocator.getLastKnownPosition();
      } catch (_) {
        // ignore，继续按下面的逻辑抛错
      }
    }

    if (pos != null) {
      return wgs84ToGcj02(LatLng(pos.latitude, pos.longitude));
    }

    if (lastError is TimeoutException) {
      throw const LocationException(
        LocationFailureReason.timeout,
        '获取定位超时，请到空旷处或开启 GPS 后重试。',
      );
    }
    throw LocationException(
      LocationFailureReason.unknown,
      '获取定位失败：${lastError ?? '未知错误'}',
    );
  }
}
