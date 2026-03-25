part of '../record.dart';

class _TipBlock extends StatelessWidget {
  final String title;
  final String body;

  const _TipBlock({required this.title, required this.body});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white.withAlpha(12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withAlpha(18)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              color: BDDesign.colorPaperWhite,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            body,
            style: TextStyle(color: Colors.white.withAlpha(176), height: 1.45),
          ),
        ],
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  final String label;
  final Color color;
  final Color backgroundColor;
  final bool isSquareDot;
  final bool compact;

  const _StatusPill({
    super.key,
    required this.label,
    required this.color,
    required this.backgroundColor,
    this.isSquareDot = false,
    this.compact = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: compact ? 10 : 12,
        vertical: compact ? 5 : 7,
      ),
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withAlpha(120)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: compact ? 7 : 8,
            height: compact ? 7 : 8,
            decoration: BoxDecoration(
              color: color,
              shape: isSquareDot ? BoxShape.rectangle : BoxShape.circle,
            ),
          ),
          SizedBox(width: compact ? 6 : 8),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: compact ? 10 : 11,
              fontWeight: FontWeight.w700,
              letterSpacing: compact ? 0.3 : 0.6,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ],
      ),
    );
  }
}

class _RecordOverlayPanel extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;

  const _RecordOverlayPanel({required this.child, required this.padding});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: BDDesign.colorInkBlack.withAlpha(216),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withAlpha(28)),
        boxShadow: [BDDesign.shadowElevated],
      ),
      child: child,
    );
  }
}
