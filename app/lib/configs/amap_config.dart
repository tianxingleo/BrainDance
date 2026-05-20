import 'package:flutter_dotenv/flutter_dotenv.dart';

class AmapConfig {
  static String get webServiceKey =>
      dotenv.env['AMAP_WEB_SERVICE_KEY']?.trim().isNotEmpty == true
          ? dotenv.env['AMAP_WEB_SERVICE_KEY']!.trim()
          : (dotenv.env['AMAP_KEY']?.trim() ?? '');

  static bool get hasWebServiceKey => webServiceKey.isNotEmpty;

  static Uri staticMapUri({
    required double latitude,
    required double longitude,
    required int zoom,
    required int width,
    required int height,
    int scale = 2,
  }) {
    return Uri.https('restapi.amap.com', '/v3/staticmap', {
      'key': webServiceKey,
      'location': '${longitude.toStringAsFixed(6)},${latitude.toStringAsFixed(6)}',
      'zoom': zoom.clamp(1, 17).toString(),
      'size': '${width.clamp(1, 1024)}*${height.clamp(1, 1024)}',
      'scale': scale.clamp(1, 2).toString(),
    });
  }
}
