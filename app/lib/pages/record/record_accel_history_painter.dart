part of '../record.dart';

class _AccelHistoryPainter extends CustomPainter {
  final List<double> samples;
  final Color color;

  const _AccelHistoryPainter({required this.samples, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final bounds = Offset.zero & size;
    final background = Paint()..color = Colors.white.withAlpha(10);
    canvas.drawRRect(
      RRect.fromRectAndRadius(bounds, const Radius.circular(10)),
      background,
    );

    double yFor(double value) {
      final normalized = (value / _kInstantSpikeAccel).clamp(0.0, 1.0);
      return size.height - (size.height * normalized);
    }

    final safeBand = Paint()..color = BDDesign.colorFadedOlive.withAlpha(26);
    final cautionBand = Paint()..color = const Color(0xFFB88746).withAlpha(22);
    final dangerBand = Paint()..color = BDDesign.colorDarkRed.withAlpha(20);

    canvas.drawRect(
      Rect.fromLTRB(
        0,
        yFor(_kIdealAccelMax),
        size.width,
        yFor(_kIdealAccelMin),
      ),
      safeBand,
    );
    canvas.drawRect(
      Rect.fromLTRB(
        0,
        yFor(_kCautionAccelMax),
        size.width,
        yFor(_kIdealAccelMax),
      ),
      cautionBand,
    );
    canvas.drawRect(
      Rect.fromLTRB(
        0,
        yFor(_kInstantSpikeAccel),
        size.width,
        yFor(_kCautionAccelMax),
      ),
      dangerBand,
    );

    final gridPaint = Paint()
      ..color = Colors.white.withAlpha(20)
      ..strokeWidth = 1;
    for (final marker in <double>[
      _kIdealAccelMax,
      _kCautionAccelMax,
      _kDangerAccelMax,
    ]) {
      final y = yFor(marker);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    if (samples.isEmpty) {
      return;
    }

    final path = Path();
    for (var i = 0; i < samples.length; i++) {
      final x = samples.length == 1
          ? size.width
          : (size.width * i) / (samples.length - 1);
      final y = yFor(samples[i]);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    final linePaint = Paint()
      ..shader = LinearGradient(
        colors: [color.withAlpha(120), color],
      ).createShader(bounds)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.4
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    canvas.drawPath(path, linePaint);
    canvas.drawCircle(
      Offset(size.width, yFor(samples.last)),
      3.5,
      Paint()..color = color,
    );
  }

  @override
  bool shouldRepaint(covariant _AccelHistoryPainter oldDelegate) {
    return oldDelegate.samples != samples || oldDelegate.color != color;
  }
}
