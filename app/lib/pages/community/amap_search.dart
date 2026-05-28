import 'package:dio/dio.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:latlong2/latlong.dart';

/// 高德地点搜索（POI）单条结果。
class AmapPoi {
  final String id;
  final String name;
  final String address;
  final LatLng location;
  final String type;
  final String cityName;
  final String district;

  const AmapPoi({
    required this.id,
    required this.name,
    required this.address,
    required this.location,
    required this.type,
    required this.cityName,
    required this.district,
  });

  /// 区域信息组合，用于结果列表副标题展示。
  String get regionLabel {
    final parts = <String>[];
    if (cityName.isNotEmpty) parts.add(cityName);
    if (district.isNotEmpty && district != cityName) parts.add(district);
    return parts.join(' · ');
  }
}

class AmapSearchException implements Exception {
  final String message;
  final String? code;
  const AmapSearchException(this.message, {this.code});

  @override
  String toString() => 'AmapSearchException($code): $message';
}

/// 高德逆地理编码结果（简化）。
class AmapRegeo {
  /// 适合作为 placeName 显示的短文案：优先 AOI/POI 名，其次 township/district。
  final String placeName;

  /// 高德 formatted_address（包含省市区+街道+门牌）。
  final String formattedAddress;

  final String cityName;
  final String district;

  const AmapRegeo({
    required this.placeName,
    required this.formattedAddress,
    required this.cityName,
    required this.district,
  });
}

/// 单例的高德文本搜索服务。
class AmapSearchService {
  AmapSearchService._internal();
  static final AmapSearchService instance = AmapSearchService._internal();

  static const String _baseUrl = 'https://restapi.amap.com';
  static const String _textSearchPath = '/v3/place/text';
  static const Duration _timeout = Duration(seconds: 8);

  Dio? _dio;

  Dio _client() {
    return _dio ??= Dio(
      BaseOptions(
        baseUrl: _baseUrl,
        connectTimeout: _timeout,
        sendTimeout: _timeout,
        receiveTimeout: _timeout,
        responseType: ResponseType.json,
      ),
    );
  }

  String _readKey() {
    final key = dotenv.env['AMAP_WEB_SERVICE_KEY']?.trim();
    if (key == null || key.isEmpty) {
      throw const AmapSearchException(
        '未配置 AMAP_WEB_SERVICE_KEY，无法调用高德地点搜索。',
      );
    }
    return key;
  }

  /// 关键字搜索。返回的 POI 顺序与高德返回保持一致。
  /// [city] 可选，传入则把搜索范围限定为该城市。
  Future<List<AmapPoi>> searchByText(
    String keywords, {
    String? city,
    int offset = 20,
    int page = 1,
    CancelToken? cancelToken,
  }) async {
    final keyword = keywords.trim();
    if (keyword.isEmpty) return const [];

    final key = _readKey();
    final params = <String, dynamic>{
      'key': key,
      'keywords': keyword,
      'offset': offset.clamp(1, 25),
      'page': page.clamp(1, 100),
      'extensions': 'base',
      'output': 'json',
    };
    if (city != null && city.isNotEmpty) {
      params['city'] = city;
      params['citylimit'] = 'true';
    }

    final Response<dynamic> resp;
    try {
      resp = await _client().get<dynamic>(
        _textSearchPath,
        queryParameters: params,
        cancelToken: cancelToken,
      );
    } on DioException catch (e) {
      if (CancelToken.isCancel(e)) rethrow;
      throw AmapSearchException(
        '网络请求失败：${e.message ?? e.type.name}',
      );
    }

    final data = resp.data;
    if (data is! Map) {
      throw const AmapSearchException('响应格式异常');
    }
    final status = data['status']?.toString();
    if (status != '1') {
      final info = data['info']?.toString() ?? '未知错误';
      final code = data['infocode']?.toString();
      throw AmapSearchException(info, code: code);
    }

    final pois = data['pois'];
    if (pois is! List) return const [];

    final results = <AmapPoi>[];
    for (final raw in pois) {
      final poi = _parsePoi(raw);
      if (poi != null) results.add(poi);
    }
    return results;
  }

  AmapPoi? _parsePoi(dynamic raw) {
    if (raw is! Map) return null;
    final location = _parseLocation(raw['location']);
    if (location == null) return null;
    return AmapPoi(
      id: _stringField(raw['id']),
      name: _stringField(raw['name']),
      address: _stringField(raw['address']),
      location: location,
      type: _stringField(raw['type']),
      cityName: _stringField(raw['cityname']),
      district: _stringField(raw['adname']),
    );
  }

  /// 逆地理编码：把 GCJ-02 坐标解析为可读地址。
  /// 注意：传入的 [point] 必须是 GCJ-02 坐标，调用方应先做转换。
  Future<AmapRegeo> regeoSearch(
    LatLng point, {
    CancelToken? cancelToken,
  }) async {
    final key = _readKey();
    final params = <String, dynamic>{
      'key': key,
      'location':
          '${point.longitude.toStringAsFixed(6)},${point.latitude.toStringAsFixed(6)}',
      'extensions': 'base',
      'output': 'json',
      'radius': 200,
    };
    final Response<dynamic> resp;
    try {
      resp = await _client().get<dynamic>(
        '/v3/geocode/regeo',
        queryParameters: params,
        cancelToken: cancelToken,
      );
    } on DioException catch (e) {
      if (CancelToken.isCancel(e)) rethrow;
      throw AmapSearchException('网络请求失败：${e.message ?? e.type.name}');
    }
    final data = resp.data;
    if (data is! Map) {
      throw const AmapSearchException('响应格式异常');
    }
    final status = data['status']?.toString();
    if (status != '1') {
      final info = data['info']?.toString() ?? '未知错误';
      final code = data['infocode']?.toString();
      throw AmapSearchException(info, code: code);
    }
    final regeocode = data['regeocode'];
    if (regeocode is! Map) {
      throw const AmapSearchException('逆地理结果为空');
    }
    final formatted = _stringField(regeocode['formatted_address']);
    final addressComp = regeocode['addressComponent'];
    String city = '';
    String district = '';
    String township = '';
    String aoiName = '';
    String poiName = '';
    if (addressComp is Map) {
      city = _stringField(addressComp['city']);
      if (city.isEmpty) city = _stringField(addressComp['province']);
      district = _stringField(addressComp['district']);
      township = _stringField(addressComp['township']);
      final aois = addressComp['aois'];
      if (aois is List && aois.isNotEmpty && aois.first is Map) {
        aoiName = _stringField((aois.first as Map)['name']);
      }
      final pois = addressComp['pois'];
      if (pois is List && pois.isNotEmpty && pois.first is Map) {
        poiName = _stringField((pois.first as Map)['name']);
      }
    }
    final placeName = [aoiName, poiName, township, district, formatted]
        .firstWhere((v) => v.isNotEmpty, orElse: () => '');
    return AmapRegeo(
      placeName: placeName,
      formattedAddress: formatted,
      cityName: city,
      district: district,
    );
  }

  static String _stringField(dynamic v) {
    if (v == null) return '';
    if (v is String) return v;
    if (v is List) return v.isEmpty ? '' : v.first.toString();
    return v.toString();
  }

  /// 高德 location 字段格式："经度,纬度"。
  static LatLng? _parseLocation(dynamic raw) {
    if (raw is! String) return null;
    final parts = raw.split(',');
    if (parts.length != 2) return null;
    final lng = double.tryParse(parts[0].trim());
    final lat = double.tryParse(parts[1].trim());
    if (lng == null || lat == null) return null;
    if (!lng.isFinite || !lat.isFinite) return null;
    if (lat.abs() > 90 || lng.abs() > 180) return null;
    return LatLng(lat, lng);
  }
}
