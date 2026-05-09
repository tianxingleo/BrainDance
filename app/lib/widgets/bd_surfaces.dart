import 'dart:ui';

import 'package:flutter/material.dart';

import '../configs/app_theme.dart';
import '../configs/motion_tokens.dart';

class BDPageBackdrop extends StatelessWidget {
  final Widget child;
  final bool darken;

  const BDPageBackdrop({super.key, required this.child, this.darken = false});

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final baseColor = isDark ? const Color(0xFF0E1013) : BDDesign.colorAshGray;

    return DecoratedBox(
      decoration: BoxDecoration(color: darken ? Colors.black : baseColor),
      child: Stack(
        children: [
          Positioned.fill(
            child: IgnorePointer(
              child: DecoratedBox(
                decoration: BoxDecoration(
                  gradient: RadialGradient(
                    center: const Alignment(-0.72, -0.86),
                    radius: 1.25,
                    colors: [
                      BDDesign.colorMutedBlue.withValues(
                        alpha: isDark ? 0.24 : 0.14,
                      ),
                      baseColor.withValues(alpha: 0.0),
                    ],
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            top: 140,
            right: -40,
            child: IgnorePointer(
              child: _BackdropOrb(
                color: BDDesign.colorFadedOlive.withValues(
                  alpha: isDark ? 0.12 : 0.09,
                ),
                size: 220,
              ),
            ),
          ),
          Positioned(
            left: -60,
            bottom: 80,
            child: IgnorePointer(
              child: _BackdropOrb(
                color: BDDesign.colorMutedBlueLight.withValues(
                  alpha: isDark ? 0.12 : 0.18,
                ),
                size: 260,
              ),
            ),
          ),
          child,
        ],
      ),
    );
  }
}

class _BackdropOrb extends StatelessWidget {
  final Color color;
  final double size;

  const _BackdropOrb({required this.color, required this.size});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: RadialGradient(
          colors: [
            color,
            color.withValues(alpha: color.a * 0.42),
            Colors.transparent,
          ],
          stops: const [0.0, 0.58, 1.0],
        ),
      ),
    );
  }
}

class BDPageHeader extends StatelessWidget {
  final String title;
  final String? subtitle;
  final Widget? trailing;
  final EdgeInsetsGeometry padding;

  const BDPageHeader({
    super.key,
    required this.title,
    this.subtitle,
    this.trailing,
    this.padding = const EdgeInsets.fromLTRB(20, 20, 20, 12),
  });

  @override
  Widget build(BuildContext context) {
    final textColor = context.isDarkMode
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final hintColor = context.isDarkMode
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return Padding(
      padding: padding,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    color: textColor,
                    fontSize: 26,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.6,
                  ),
                ),
                if (subtitle != null) ...[
                  const SizedBox(height: 6),
                  Text(
                    subtitle!,
                    style: TextStyle(
                      color: hintColor,
                      fontSize: 13.5,
                      height: 1.4,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ],
            ),
          ),
          if (trailing != null) ...[const SizedBox(width: 12), trailing!],
        ],
      ),
    );
  }
}

class BDPanelCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final BorderRadius? borderRadius;
  final bool glass;

  const BDPanelCard({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.borderRadius,
    this.glass = false,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    if (glass) {
      return BDGlassSurface(
        margin: margin,
        padding: padding,
        borderRadius: borderRadius ?? BDDesign.radiusLarge,
        variant: BDGlassVariant.panel,
        child: child,
      );
    }

    final background = isDark
        ? AppTheme.darkSurface.withValues(alpha: 0.94)
        : BDDesign.colorPaperWhite.withValues(alpha: 0.94);
    final borderColor = isDark
        ? Colors.white.withValues(alpha: 0.08)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.10);

    return Container(
      margin: margin,
      padding: padding,
      decoration: BoxDecoration(
        color: background,
        borderRadius: borderRadius ?? BDDesign.radiusLarge,
        border: Border.all(color: borderColor),
        boxShadow: [isDark ? BDDesign.shadowLight : BDDesign.shadowElevated],
      ),
      child: child,
    );
  }
}

enum BDGlassVariant { panel, floating }

class BDGlassSurface extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final BorderRadius borderRadius;
  final BDGlassVariant variant;
  final double? blurSigma;
  final Color? tintColor;
  final Color? borderColor;
  final List<BoxShadow>? shadows;

  const BDGlassSurface({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.borderRadius = const BorderRadius.all(Radius.circular(24)),
    this.variant = BDGlassVariant.panel,
    this.blurSigma,
    this.tintColor,
    this.borderColor,
    this.shadows,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final effectiveBlur =
        blurSigma ??
        switch (variant) {
          BDGlassVariant.panel => 18.0,
          BDGlassVariant.floating => 24.0,
        };
    final effectiveTint =
        tintColor ??
        (isDark
            ? AppTheme.darkSurface.withValues(
                alpha: variant == BDGlassVariant.floating ? 0.56 : 0.72,
              )
            : BDDesign.colorPaperWhite.withValues(
                alpha: variant == BDGlassVariant.floating ? 0.54 : 0.66,
              ));
    final effectiveBorder =
        borderColor ??
        (isDark
            ? Colors.white.withValues(alpha: 0.09)
            : BDDesign.colorMutedBlue.withValues(alpha: 0.12));
    final effectiveShadows =
        shadows ??
        [
          BoxShadow(
            color: Colors.black.withValues(
              alpha: isDark
                  ? (variant == BDGlassVariant.floating ? 0.24 : 0.18)
                  : (variant == BDGlassVariant.floating ? 0.06 : 0.04),
            ),
            blurRadius: variant == BDGlassVariant.floating ? 28 : 22,
            offset: Offset(0, variant == BDGlassVariant.floating ? 8 : 6),
          ),
        ];

    return RepaintBoundary(
      child: Container(
        margin: margin,
        decoration: BoxDecoration(
          borderRadius: borderRadius,
          boxShadow: effectiveShadows,
        ),
        child: ClipRRect(
          borderRadius: borderRadius,
          child: BackdropFilter(
            filter: ImageFilter.blur(
              sigmaX: effectiveBlur,
              sigmaY: effectiveBlur,
            ),
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: effectiveTint,
                borderRadius: borderRadius,
                border: Border.all(color: effectiveBorder),
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    Colors.white.withValues(alpha: isDark ? 0.08 : 0.28),
                    Colors.white.withValues(alpha: isDark ? 0.02 : 0.08),
                  ],
                ),
              ),
              child: padding == null
                  ? child
                  : Padding(padding: padding!, child: child),
            ),
          ),
        ),
      ),
    );
  }
}

class BDStatusPill extends StatelessWidget {
  final String label;
  final IconData? icon;
  final Color? color;

  const BDStatusPill({super.key, required this.label, this.icon, this.color});

  @override
  Widget build(BuildContext context) {
    final pillColor = color ?? BDDesign.colorMutedBlue;

    return AnimatedContainer(
      duration: BDMotion.durationNormal,
      curve: BDMotion.curveFluid,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: pillColor.withValues(alpha: 0.11),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: pillColor.withValues(alpha: 0.18)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 14, color: pillColor),
            const SizedBox(width: 6),
          ],
          Flexible(
            child: Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              softWrap: false,
              style: TextStyle(
                color: pillColor,
                fontSize: 12.5,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.1,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
