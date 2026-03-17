part of '../generate.dart';

class _GenerateSectionHeading extends StatelessWidget {
  final String title;
  final String description;

  const _GenerateSectionHeading({
    required this.title,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          description,
          style: TextStyle(
            fontSize: 13,
            height: 1.45,
            color: isDark
                ? Colors.white.withValues(alpha: 0.62)
                : BDDesign.colorMutedBlue,
          ),
        ),
      ],
    );
  }
}

class _GenerateMetric extends StatelessWidget {
  final String label;
  final String value;

  const _GenerateMetric({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: isDark
                ? Colors.white.withValues(alpha: 0.58)
                : BDDesign.colorMutedBlue,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          value,
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w700,
            color: isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack,
          ),
        ),
      ],
    );
  }
}
