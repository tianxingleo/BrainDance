import 'dart:math' as math;

abstract class LocalTextEmbedder {
  int get dimensions;

  List<double> embed(String text);
}

class HashingTextEmbedder implements LocalTextEmbedder {
  HashingTextEmbedder({this.dimensions = 192});

  @override
  final int dimensions;

  static final RegExp _wordPattern = RegExp(r'[a-z0-9]+|[\u4e00-\u9fff]');

  @override
  List<double> embed(String text) {
    final normalized = _normalize(text);
    final vector = List<double>.filled(dimensions, 0);
    if (normalized.isEmpty) {
      return vector;
    }

    final compact = normalized.replaceAll(' ', '');
    for (final token in _wordPattern.allMatches(normalized)) {
      final value = token.group(0);
      if (value == null || value.isEmpty) {
        continue;
      }
      _accumulate(vector, value, 1.0);
    }

    for (final gram in _charNgrams(compact, 2)) {
      _accumulate(vector, gram, 0.45);
    }
    for (final gram in _charNgrams(compact, 3)) {
      _accumulate(vector, gram, 0.25);
    }

    final norm = math.sqrt(
      vector.fold<double>(0, (sum, value) => sum + value * value),
    );
    if (norm <= 1e-9) {
      return vector;
    }

    for (var i = 0; i < vector.length; i++) {
      vector[i] = vector[i] / norm;
    }
    return vector;
  }

  void _accumulate(List<double> vector, String token, double weight) {
    final hash = _fnv1a32(token);
    final index = hash % dimensions;
    final sign = ((hash >> 1) & 1) == 0 ? 1.0 : -1.0;
    vector[index] += sign * weight;
  }

  Iterable<String> _charNgrams(String text, int n) sync* {
    if (text.length < n) {
      return;
    }
    for (var i = 0; i <= text.length - n; i++) {
      yield text.substring(i, i + n);
    }
  }

  String _normalize(String input) {
    return input
        .toLowerCase()
        .replaceAll(RegExp(r'\s+'), ' ')
        .replaceAll(RegExp(r'[^\p{L}\p{N}\u4e00-\u9fff ]', unicode: true), ' ')
        .trim();
  }

  int _fnv1a32(String input) {
    const int offset = 0x811c9dc5;
    const int prime = 0x01000193;
    var hash = offset;
    for (final codeUnit in input.codeUnits) {
      hash ^= codeUnit;
      hash = (hash * prime) & 0xffffffff;
    }
    return hash;
  }
}
