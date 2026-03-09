class SpaceCapture {
  final String id;
  final String sceneId;
  final DateTime capturedAt;
  final String status;
  final String? modelUrl;
  final List<double>? alignmentMatrix;
  final double? alignmentScore;

  SpaceCapture({
    required this.id,
    required this.sceneId,
    required this.capturedAt,
    required this.status,
    required this.modelUrl,
    required this.alignmentMatrix,
    required this.alignmentScore,
  });

  factory SpaceCapture.fromJson(Map<String, dynamic> json) {
    final matrixRaw = json['alignment_matrix'];
    List<double>? matrix;
    if (matrixRaw is List) {
      matrix = matrixRaw.map((e) => (e as num).toDouble()).toList();
    }

    return SpaceCapture(
      id: json['capture_id']?.toString() ?? '',
      sceneId: json['scene_id']?.toString() ?? 'unknown_scene',
      capturedAt: DateTime.tryParse(json['captured_at']?.toString() ?? '') ?? DateTime.now(),
      status: json['status']?.toString() ?? 'unknown',
      modelUrl: json['model_url']?.toString(),
      alignmentMatrix: matrix,
      alignmentScore: (json['alignment_score'] as num?)?.toDouble(),
    );
  }
}

class TimePeelingPayload {
  final String baseModelUrl;
  final String overlayModelUrl;
  final List<double> overlayMatrix;
  final double alpha;
  final List<double>? pose;

  TimePeelingPayload({
    required this.baseModelUrl,
    required this.overlayModelUrl,
    required this.overlayMatrix,
    required this.alpha,
    required this.pose,
  });

  factory TimePeelingPayload.fromJson(Map<String, dynamic> json) {
    List<double> parseMatrix(dynamic value) {
      if (value is List) {
        return value.map((e) => (e as num).toDouble()).toList();
      }
      return const [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ];
    }

    List<double>? parsePose(dynamic value) {
      if (value is List) {
        return value.map((e) => (e as num).toDouble()).toList();
      }
      return null;
    }

    return TimePeelingPayload(
      baseModelUrl: json['base_model']?.toString() ?? '',
      overlayModelUrl: json['overlay_model']?.toString() ?? '',
      overlayMatrix: parseMatrix(json['overlay_alignment_matrix']),
      alpha: (json['default_alpha'] as num?)?.toDouble() ?? 0.5,
      pose: parsePose(json['initial_pose']),
    );
  }

  Map<String, dynamic> toWebPayload() {
    return {
      'base': baseModelUrl,
      'overlay': overlayModelUrl,
      'matrix': overlayMatrix,
      'alpha': alpha,
      if (pose != null) 'pose': pose,
    };
  }
}
