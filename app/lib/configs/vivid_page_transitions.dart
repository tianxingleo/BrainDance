import 'package:flutter/material.dart';

/// 灵动舒适的页面过渡动画 (类现代 iOS / Material 3 混合风格)
/// 拥有 3D 景深感，包含透明度、缩放、视差位移的多重组合
class VividPageTransitionsBuilder extends PageTransitionsBuilder {
  const VividPageTransitionsBuilder();

  @override
  Widget buildTransitions<T>(
    PageRoute<T> route,
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
    Widget child,
  ) {
    // 主进入动画曲线 (新页面入场)：快速进入，缓慢定格，给人丝滑感
    final fastOutSlowIn = CurvedAnimation(
      parent: animation,
      curve: const Cubic(0.05, 0.7, 0.1, 1.0), // 定制的超平滑弹出曲线 (类似苹果 Spring 动画)
      reverseCurve: const Cubic(0.3, 0.0, 0.8, 0.15), // 退出曲线收缩快些
    );

    // 副轴动画曲线 (旧页面退场)：当有新页面盖在上面时，当前页面的动画
    final secondaryFastOutSlowIn = CurvedAnimation(
      parent: secondaryAnimation,
      curve: const Cubic(0.05, 0.7, 0.1, 1.0),
      reverseCurve: const Cubic(0.3, 0.0, 0.8, 0.15),
    );

    // 被盖住（向后退场）的动画设定
    final secondarySlide = SlideTransition(
      position: Tween<Offset>(
        begin: Offset.zero,
        end: const Offset(-0.2, 0.0), // 取代系统的直接覆盖，旧页面优雅滑向左侧 20%
      ).animate(secondaryFastOutSlowIn),
      child: ScaleTransition(
        scale: Tween<double>(
          begin: 1.0,
          end: 0.92,
        ).animate(secondaryFastOutSlowIn), // 缩小到后台
        child: FadeTransition(
          opacity: Tween<double>(
            begin: 1.0,
            end: 0.4,
          ).animate(secondaryFastOutSlowIn), // 给底部压暗效果
          child: child,
        ),
      ),
    );

    // 新页面入场（推向前方）的动画设定
    return SlideTransition(
      position: Tween<Offset>(
        begin: const Offset(0.35, 0.0), // 新页面从右侧 35% 位置滑入（比起 100% 更加平稳短促）
        end: Offset.zero,
      ).animate(fastOutSlowIn),
      child: FadeTransition(
        opacity: Tween<double>(
          begin: 0.0,
          end: 1.0,
        ).animate(fastOutSlowIn), // 淡入
        child: ScaleTransition(
          scale: Tween<double>(
            begin: 0.95,
            end: 1.0,
          ).animate(fastOutSlowIn), // 略微放大呈现 3D 感
          child: secondarySlide,
        ),
      ),
    );
  }
}
