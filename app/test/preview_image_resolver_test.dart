import 'package:braindance/services/preview_image_resolver.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('prefers meta preview_webp_path and keeps jpg fallback', () {
    final paths = resolvePreviewImagePaths({
      'preview_img_path': 'u/scene/output/preview.jpg',
      'meta_info': {'preview_webp_path': 'u/scene/output/preview.webp'},
    }, normalize: (raw) => 'public/$raw');

    expect(paths.primary, 'public/u/scene/output/preview.webp');
    expect(paths.fallback, 'public/u/scene/output/preview.jpg');
  });

  test('uses jpg when webp preview is absent', () {
    final paths = resolvePreviewImagePaths({
      'preview_img_path': 'u/scene/output/preview.jpg',
    });

    expect(paths.primary, 'u/scene/output/preview.jpg');
    expect(paths.fallback, isNull);
  });

  test('materializes normalized webp path from meta_info', () {
    final model = <String, dynamic>{
      'preview_img_path': 'u/scene/output/preview.jpg',
      'meta_info': {'preview_webp_path': 'u/scene/output/preview.webp'},
    };

    materializePreviewWebpPath(model, normalize: (raw) => 'public/$raw');

    expect(model['preview_webp_path'], 'public/u/scene/output/preview.webp');
  });
}
