import 'package:flutter/material.dart';

/// BrainDance 的全局动效资产 (Motion Tokens)
/// 根据 [BrainDance_Design_Rules.md] 设定，采用慢半拍、有物理真实感的速度曲线。
class BDMotion {
  // === 持续时间 (Durations) ===
  /// 小交互 (按钮按下、选中状态)：120-180ms
  static const Duration durationFast = Duration(milliseconds: 150);

  /// 卡片浮现、普通状态切换：220-280ms
  static const Duration durationNormal = Duration(milliseconds: 250);

  /// 页面切换、大型组件入场：320-420ms
  static const Duration durationSlow = Duration(milliseconds: 380);

  // === 动画曲线 (Curves) ===
  /// 进入视口 (Decelerate)
  static const Curve curveEnter = Curves.easeOutCubic;

  /// 离开视口 (Accelerate)
  static const Curve curveExit = Curves.easeInCubic;

  /// 漂浮、跟手、状态自然流动 (极具粘滞感)
  static const Curve curveFluid = Curves.easeOutQuart;

  /// 呼吸/脉冲动画用 (对称起伏)
  static const Curve curveBreathe = Curves.easeInOutSine;
}

/// BrainDance 的全局设计资产 (Design Tokens)
class BDDesign {
  // === 颜色系统 (Colors) ===
  /// 略暖的纸白 (背景)
  static const Color colorPaperWhite = Color(0xFFF9F9F8);

  /// 纯正的石灰 (第二背景)
  static const Color colorAshGray = Color(0xFFEDEDEA);

  /// 墨黑 (主文本)
  static const Color colorInkBlack = Color(0xFF1E1E20);

  /// 钝蓝灰 (主强调色，取代荧光蓝)
  static const Color colorMutedBlue = Color(0xFF6B7A8F);
  static const Color colorMutedBlueLight = Color(0xFFE4E8ED);

  /// 暗红棕 (故障、危险状态)
  static const Color colorDarkRed = Color(0xFF8B4747);

  /// 褪色橄榄绿 (成功、进行中状态)
  static const Color colorFadedOlive = Color(0xFF6D8260);

  // === 形状与阴影 (Shapes & Shadows) ===
  static final BorderRadius radiusSmall = BorderRadius.circular(12.0);
  static final BorderRadius radiusNormal = BorderRadius.circular(20.0);
  static final BorderRadius radiusLarge = BorderRadius.circular(28.0);

  /// 极度克制的阴影
  static final BoxShadow shadowLight = BoxShadow(
    color: const Color(0xFF000000).withOpacity(0.03),
    blurRadius: 16.0,
    offset: const Offset(0, 4),
  );

  static final BoxShadow shadowElevated = BoxShadow(
    color: const Color(0xFF000000).withOpacity(0.06),
    blurRadius: 24.0,
    offset: const Offset(0, 8),
  );
}
