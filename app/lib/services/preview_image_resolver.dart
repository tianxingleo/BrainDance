class PreviewImagePaths {
  final String? primary;
  final String? fallback;

  const PreviewImagePaths({this.primary, this.fallback});

  bool get hasAny =>
      primary != null && primary!.isNotEmpty ||
      fallback != null && fallback!.isNotEmpty;
}

typedef PreviewPathNormalizer = String Function(String raw);

String? readPreviewWebpPath(Map<String, dynamic> model) {
  final direct = _readString(model['preview_webp_path']);
  if (direct != null) return direct;

  final meta = model['meta_info'];
  if (meta is Map) {
    return _readString(meta['preview_webp_path']) ??
        _readString(meta['previewWebpPath']);
  }
  return null;
}

String? readPreviewJpgPath(Map<String, dynamic> model) {
  return _readString(model['preview_img_path']);
}

void materializePreviewWebpPath(
  Map<String, dynamic> model, {
  PreviewPathNormalizer? normalize,
}) {
  final webp = readPreviewWebpPath(model);
  if (webp == null) return;
  model['preview_webp_path'] = normalize == null ? webp : normalize(webp);
}

PreviewImagePaths resolvePreviewImagePaths(
  Map<String, dynamic> model, {
  PreviewPathNormalizer? normalize,
}) {
  String? normalizePath(String? raw) {
    if (raw == null || raw.isEmpty) return null;
    final value = normalize == null ? raw : normalize(raw);
    return value.trim().isEmpty ? null : value;
  }

  final webp = normalizePath(readPreviewWebpPath(model));
  final jpg = normalizePath(readPreviewJpgPath(model));

  if (webp == null) {
    return PreviewImagePaths(primary: jpg);
  }
  if (jpg == null || jpg == webp) {
    return PreviewImagePaths(primary: webp);
  }
  return PreviewImagePaths(primary: webp, fallback: jpg);
}

String? _readString(Object? value) {
  final text = value?.toString().trim() ?? '';
  return text.isEmpty ? null : text;
}
