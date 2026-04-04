part of '../record.dart';

class _SimpleMotionGuidanceCard extends StatelessWidget {
  final double motionMeter;
  final _MotionState motionState;
  final String motionHint;
  final String motionDetail;

  const _SimpleMotionGuidanceCard({
    required this.motionMeter,
    required this.motionState,
    required this.motionHint,
    required this.motionDetail,
  });

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final cardWidth = (size.width * 0.42).clamp(170.0, 220.0);
    final progressValue = switch (motionState) {
      _MotionState.steady => 0.18,
      _MotionState.ideal => 0.42 + (motionMeter / _kIdealAccelMax) * 0.18,
      _MotionState.caution =>
        0.68 +
            ((motionMeter - _kIdealAccelMax) /
                    (_kCautionAccelMax - _kIdealAccelMax)) *
                0.18,
      _MotionState.danger =>
        0.9 +
            ((motionMeter - _kCautionAccelMax) /
                    (_kInstantSpikeAccel - _kCautionAccelMax)) *
                0.1,
    };
    final normalizedAccel = progressValue.clamp(0.0, 1.0);
    final guideColor = switch (motionState) {
      _MotionState.steady => BDDesign.colorMutedBlue,
      _MotionState.ideal => BDDesign.colorFadedOlive,
      _MotionState.caution => const Color(0xFFB88746),
      _MotionState.danger => BDDesign.colorDarkRed,
    };
    final stateLabel = switch (motionState) {
      _MotionState.steady => textLocalize('reco_state_steady'),
      _MotionState.ideal => textLocalize('reco_state_ideal'),
      _MotionState.caution => textLocalize('reco_state_caution'),
      _MotionState.danger => textLocalize('reco_state_danger'),
    };

    return Container(
      width: cardWidth,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      decoration: BoxDecoration(
        color: BDDesign.colorInkBlack.withAlpha(216),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: guideColor.withAlpha(160)),
        boxShadow: [BDDesign.shadowElevated],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 9,
                height: 9,
                decoration: BoxDecoration(
                  color: guideColor,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  textLocalize('sensor_on'),
                  style: const TextStyle(
                    color: BDDesign.colorPaperWhite,
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
                decoration: BoxDecoration(
                  color: guideColor.withAlpha(36),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  stateLabel,
                  style: TextStyle(
                    color: guideColor,
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          TweenAnimationBuilder<double>(
            tween: Tween<double>(end: normalizedAccel),
            duration: const Duration(milliseconds: 260),
            curve: Curves.easeOutCubic,
            builder: (context, value, _) {
              return ClipRRect(
                borderRadius: BorderRadius.circular(999),
                child: LinearProgressIndicator(
                  minHeight: 9,
                  value: value,
                  backgroundColor: Colors.white.withAlpha(28),
                  valueColor: AlwaysStoppedAnimation<Color>(guideColor),
                ),
              );
            },
          ),
          const SizedBox(height: 8),
          Text(
            motionHint,
            style: TextStyle(
              color: Colors.white.withAlpha(210),
              fontSize: 11,
              height: 1.35,
            ),
          ),
          const SizedBox(height: 5),
          Text(
            motionDetail,
            style: TextStyle(
              color: BDDesign.colorAshGray.withAlpha(220),
              fontSize: 10,
              height: 1.3,
            ),
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}
