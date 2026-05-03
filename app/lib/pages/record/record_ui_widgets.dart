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

// ===== Accelerometer Warning Banner =====
class _AccelWarningBanner extends ConsumerStatefulWidget {
  const _AccelWarningBanner();

  @override
  ConsumerState<_AccelWarningBanner> createState() =>
      _AccelWarningBannerState();
}

class _AccelWarningBannerState extends ConsumerState<_AccelWarningBanner>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<Offset> _slide;
  Timer? _dismissTimer;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      duration: const Duration(milliseconds: 350),
      vsync: this,
    );
    _slide = Tween<Offset>(
      begin: const Offset(0, -1.2),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeOutCubic));
    _ctrl.addStatusListener((s) {
      if (s == AnimationStatus.dismissed) {
        ref.read(showAccelBannerProvider.notifier).state = false;
      }
    });
  }

  @override
  void dispose() {
    _dismissTimer?.cancel();
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    ref.listen(showAccelBannerProvider, (prev, next) {
      if (next) {
        _dismissTimer?.cancel();
        _ctrl.forward();
        _dismissTimer = Timer(const Duration(seconds: 3), () {
          if (mounted) _ctrl.reverse();
        });
      }
    });

    return Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: AnimatedBuilder(
        animation: _ctrl,
        builder: (context, child) {
          if (_ctrl.isDismissed) return const SizedBox.shrink();

          return SafeArea(
            child: SlideTransition(
              position: _slide,
              child: GestureDetector(
                onTap: () {
                  _dismissTimer?.cancel();
                  _ctrl.reverse();
                },
                child: Container(
                  margin: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 8,
                  ),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 12,
                  ),
                  decoration: BoxDecoration(
                    color: const Color(0xFF2A2A30),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.orange.withAlpha(80)),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withAlpha(40),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.orange.withAlpha(25),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: const Icon(
                          Icons.speed,
                          color: Colors.orange,
                          size: 20,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          textLocalize('reco_accel_warning'),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

// ===== Save Fail Bubble (middle-lower floating) =====
class _SaveFailBubble extends ConsumerStatefulWidget {
  const _SaveFailBubble();

  @override
  ConsumerState<_SaveFailBubble> createState() => _SaveFailBubbleState();
}

class _SaveFailBubbleState extends ConsumerState<_SaveFailBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _fade;
  late final Animation<Offset> _slide;
  Timer? _dismissTimer;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      duration: const Duration(milliseconds: 280),
      vsync: this,
    );
    _fade = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _ctrl, curve: Curves.easeOutCubic),
    );
    _slide = Tween<Offset>(
      begin: const Offset(0, 0.15),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeOutCubic));
    _ctrl.addStatusListener((s) {
      if (s == AnimationStatus.dismissed) {
        ref.read(saveFailBubbleProvider.notifier).state = null;
      }
    });
  }

  @override
  void dispose() {
    _dismissTimer?.cancel();
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final message = ref.watch(saveFailBubbleProvider);

    ref.listen(saveFailBubbleProvider, (prev, next) {
      if (next != null) {
        _dismissTimer?.cancel();
        _ctrl.forward();
        _dismissTimer = Timer(const Duration(seconds: 4), () {
          if (mounted) _ctrl.reverse();
        });
      }
    });

    return Positioned.fill(
      child: AnimatedBuilder(
        animation: _ctrl,
        builder: (context, child) {
          if (_ctrl.isDismissed) return const SizedBox.shrink();
          if (message == null) return const SizedBox.shrink();

          return Align(
            alignment: const Alignment(0, 0.30),
            child: FadeTransition(
              opacity: _fade,
              child: SlideTransition(
                position: _slide,
                child: GestureDetector(
                  onTap: () {
                    _dismissTimer?.cancel();
                    _ctrl.reverse();
                  },
                  child: Container(
                    margin: const EdgeInsets.symmetric(horizontal: 32),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 18,
                      vertical: 12,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xE6282828),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: Colors.white.withAlpha(18)),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.error_outline_rounded,
                          color: Colors.orange.withAlpha(200),
                          size: 20,
                        ),
                        const SizedBox(width: 10),
                        Flexible(
                          child: Text(
                            message,
                            style: TextStyle(
                              color: Colors.white.withAlpha(220),
                              fontSize: 13,
                              fontWeight: FontWeight.w500,
                              height: 1.35,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

// ===== Recording Too Short Bubble (center) =====
class _TooShortBubble extends ConsumerStatefulWidget {
  const _TooShortBubble();

  @override
  ConsumerState<_TooShortBubble> createState() => _TooShortBubbleState();
}

class _TooShortBubbleState extends ConsumerState<_TooShortBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _fade;
  late final Animation<double> _scale;
  Timer? _dismissTimer;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      duration: const Duration(milliseconds: 250),
      vsync: this,
    );
    _fade = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _ctrl, curve: Curves.easeOutCubic),
    );
    _scale = Tween<double>(begin: 0.88, end: 1).animate(
      CurvedAnimation(parent: _ctrl, curve: Curves.easeOutCubic),
    );
    _ctrl.addStatusListener((s) {
      if (s == AnimationStatus.dismissed) {
        ref.read(showTooShortBubbleProvider.notifier).state = false;
      }
    });
  }

  @override
  void dispose() {
    _dismissTimer?.cancel();
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    ref.listen(showTooShortBubbleProvider, (prev, next) {
      if (next) {
        _dismissTimer?.cancel();
        _ctrl.forward();
        _dismissTimer = Timer(const Duration(seconds: 3), () {
          if (mounted) _ctrl.reverse();
        });
      }
    });

    return Positioned.fill(
      child: AnimatedBuilder(
        animation: _ctrl,
        builder: (context, child) {
          if (_ctrl.isDismissed) return const SizedBox.shrink();

          return Align(
            alignment: Alignment.center,
            child: FadeTransition(
              opacity: _fade,
              child: ScaleTransition(
                scale: _scale,
                child: GestureDetector(
                  onTap: () {
                    _dismissTimer?.cancel();
                    _ctrl.reverse();
                  },
                  child: Container(
                    margin: const EdgeInsets.symmetric(horizontal: 32),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 13,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xE6282828),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: Colors.white.withAlpha(18)),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.info_outline_rounded,
                          color: Colors.white.withAlpha(180),
                          size: 20,
                        ),
                        const SizedBox(width: 10),
                        Flexible(
                          child: Text(
                            textLocalize('reco_record_too_short'),
                            style: TextStyle(
                              color: Colors.white.withAlpha(220),
                              fontSize: 13,
                              fontWeight: FontWeight.w500,
                              height: 1.35,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
