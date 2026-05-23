import 'dart:math' as math;

import 'package:latlong2/latlong.dart';

/// 国测局加密 (GCJ-02) 与 GPS (WGS-84) 之间的偏移转换。
///
/// 高德 Web 服务（搜索 / 逆地理）以及高德地图瓦片均使用 GCJ-02 坐标系，
/// 而 `geolocator` 直接读取系统 GPS，得到的是 WGS-84，需要做一次纠偏，
/// 否则在地图上会出现 50–500m 的偏移。
///
/// 算法：经典的国测局加密算法（参考公开实现），仅适用于中国大陆区域。
/// 对中国大陆境外的坐标，按惯例直接返回原值，因为高德对境外不做偏移。

const double _piA = 3.14159265358979324;
const double _earthA = 6378245.0; // 长半轴
const double _earthEE = 0.00669342162296594323; // 偏心率平方

/// 将 WGS-84 坐标转换为 GCJ-02 坐标。境外原样返回。
LatLng wgs84ToGcj02(LatLng src) {
  if (_outsideChina(src.latitude, src.longitude)) return src;
  final dLat = _transformLat(src.longitude - 105.0, src.latitude - 35.0);
  final dLon = _transformLon(src.longitude - 105.0, src.latitude - 35.0);
  final radLat = src.latitude / 180.0 * _piA;
  final magic = math.sin(radLat);
  final m = 1 - _earthEE * magic * magic;
  final sqrtM = math.sqrt(m);
  final mgcjLat = src.latitude +
      (dLat * 180.0) /
          ((_earthA * (1 - _earthEE)) / (m * sqrtM) * _piA);
  final mgcjLon = src.longitude +
      (dLon * 180.0) / (_earthA / sqrtM * math.cos(radLat) * _piA);
  return LatLng(mgcjLat, mgcjLon);
}

/// 中国大陆地理边界的快速判断（粗略，足够用于"是否需要纠偏"的决策）。
bool _outsideChina(double lat, double lon) {
  if (lon < 72.004 || lon > 137.8347) return true;
  if (lat < 0.8293 || lat > 55.8271) return true;
  return false;
}

double _transformLat(double x, double y) {
  double ret = -100.0 +
      2.0 * x +
      3.0 * y +
      0.2 * y * y +
      0.1 * x * y +
      0.2 * math.sqrt(x.abs());
  ret += (20.0 * math.sin(6.0 * x * _piA) +
          20.0 * math.sin(2.0 * x * _piA)) *
      2.0 /
      3.0;
  ret += (20.0 * math.sin(y * _piA) +
          40.0 * math.sin(y / 3.0 * _piA)) *
      2.0 /
      3.0;
  ret += (160.0 * math.sin(y / 12.0 * _piA) +
          320.0 * math.sin(y * _piA / 30.0)) *
      2.0 /
      3.0;
  return ret;
}

double _transformLon(double x, double y) {
  double ret = 300.0 +
      x +
      2.0 * y +
      0.1 * x * x +
      0.1 * x * y +
      0.1 * math.sqrt(x.abs());
  ret += (20.0 * math.sin(6.0 * x * _piA) +
          20.0 * math.sin(2.0 * x * _piA)) *
      2.0 /
      3.0;
  ret += (20.0 * math.sin(x * _piA) +
          40.0 * math.sin(x / 3.0 * _piA)) *
      2.0 /
      3.0;
  ret += (150.0 * math.sin(x / 12.0 * _piA) +
          300.0 * math.sin(x / 30.0 * _piA)) *
      2.0 /
      3.0;
  return ret;
}
