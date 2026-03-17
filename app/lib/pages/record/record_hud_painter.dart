import 'package:flutter/material.dart';
import '../../configs/motion_tokens.dart';

/// 绘制带有“仪器感”的四个取景框角点
/// 如果 [isWarning] 为 true，边框会呈现暗红棕色的急促闪烁
class RecordHUDPainter extends CustomPainter {
  final bool isWarning;
  final bool isCaution;
  final double motionValue;
  final Animation<double> animation;

  RecordHUDPainter({
    required this.isWarning,
    required this.isCaution,
    required this.motionValue,
    required this.animation,
  }) : super(repaint: animation);

  @override
  void paint(Canvas canvas, Size size) {
    final cautionColor = const Color(0xFFB88746);
    final accentColor = isWarning
        ? BDDesign.colorDarkRed
        : isCaution
        ? cautionColor
        : Colors.white;
    final paint = Paint()
      ..color = accentColor.withValues(
        alpha: isWarning
            ? 0.8 + 0.2 * animation.value
            : isCaution
            ? 0.58 + 0.18 * animation.value
            : 0.5 + 0.3 * animation.value,
      )
      ..style = PaintingStyle.stroke
      ..strokeWidth = isWarning
          ? 3.0
          : isCaution
          ? 2.4
          : 2.0;

    final double bracketLength = 40.0;
    final double padding = 32.0;

    final left = padding;
    final top = padding * 3; // 避开顶部 UI
    final right = size.width - padding;
    final bottom = size.height - padding * 5; // 避开底部按钮

    // 左上角
    _drawBracket(canvas, paint, Offset(left, top), bracketLength, 1, 1);
    // 右上角
    _drawBracket(canvas, paint, Offset(right, top), bracketLength, -1, 1);
    // 左下角
    _drawBracket(canvas, paint, Offset(left, bottom), bracketLength, 1, -1);
    // 右下角
    _drawBracket(canvas, paint, Offset(right, bottom), bracketLength, -1, -1);

    // 绘制中央准星
    final centerPaint = Paint()
      ..color = accentColor.withValues(alpha: isWarning ? 0.6 : 0.34)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    final center = Offset(size.width / 2, size.height / 2);
    canvas.drawLine(
      center.translate(-10, 0),
      center.translate(10, 0),
      centerPaint,
    );
    canvas.drawLine(
      center.translate(0, -10),
      center.translate(0, 10),
      centerPaint,
    );
    canvas.drawCircle(center, 4, centerPaint);

    final normalizedMotion = (motionValue / 2.6).clamp(0.0, 1.0);
    final ringRadius = 26 + (16 * normalizedMotion) + (4 * animation.value);
    final ringPaint = Paint()
      ..color = accentColor.withValues(alpha: 0.18 + normalizedMotion * 0.22)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.4;
    canvas.drawCircle(center, ringRadius, ringPaint);
  }

  void _drawBracket(
    Canvas canvas,
    Paint paint,
    Offset corner,
    double length,
    double dx,
    double dy,
  ) {
    final path = Path();
    path.moveTo(corner.dx + length * dx, corner.dy);
    path.lineTo(corner.dx, corner.dy);
    path.lineTo(corner.dx, corner.dy + length * dy);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant RecordHUDPainter oldDelegate) {
    return oldDelegate.isWarning != isWarning ||
        oldDelegate.isCaution != isCaution ||
        oldDelegate.motionValue != motionValue ||
        oldDelegate.animation != animation;
  }
}
