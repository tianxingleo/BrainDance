import 'package:braindance/configs/motion_tokens.dart';
import 'package:flutter/material.dart';

/// 方向性 tab/页面切换容器。
/// 索引越大 = 越靠右；新页面从其所在方向进场，旧页面向反方向退场。
/// 兼容有界和无界父约束（如 ScrollView 内部）。
class BDTabSwitcher extends StatefulWidget {
  final int index;
  final List<Widget> children;
  final Duration duration;

  const BDTabSwitcher({
    super.key,
    required this.index,
    required this.children,
    this.duration = BDMotion.durationNormal,
  });

  @override
  State<BDTabSwitcher> createState() => _BDTabSwitcherState();
}

class _BDTabSwitcherState extends State<BDTabSwitcher>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late int _currentIndex;
  late int _previousIndex;
  bool _isAnimating = false;

  @override
  void initState() {
    super.initState();
    _currentIndex = widget.index;
    _previousIndex = widget.index;
    _ctrl = AnimationController(vsync: this, duration: widget.duration)
      ..addStatusListener((status) {
        if (status == AnimationStatus.completed && mounted) {
          setState(() {
            _isAnimating = false;
            _previousIndex = _currentIndex;
          });
        }
      });
  }

  @override
  void didUpdateWidget(BDTabSwitcher old) {
    super.didUpdateWidget(old);
    if (old.index != widget.index) {
      _previousIndex = _currentIndex;
      _currentIndex = widget.index;
      _isAnimating = true;
      _ctrl.forward(from: 0);
    }
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final int direction = _currentIndex > _previousIndex ? 1 : -1;

    if (!_isAnimating) {
      // 静止时直接渲染，不加任何约束包装
      return widget.children[_currentIndex];
    }

    // 动画中：用 LayoutBuilder 获取父约束，若有界则用 Stack+Positioned.fill，
    // 若无界则用 ClipRect+Stack 让两页面按入场页高度撑开
    return LayoutBuilder(
      builder: (context, constraints) {
        final bool bounded = constraints.hasBoundedHeight;

        return AnimatedBuilder(
          animation: _ctrl,
          builder: (context, _) {
            final double t = Curves.easeInOutCubic.transform(_ctrl.value);
            final Offset enterOffset = Offset(direction * (1.0 - t), 0);
            final Offset leaveOffset = Offset(-direction * t, 0);

            if (bounded) {
              return Stack(
                children: [
                  Positioned.fill(
                    child: IgnorePointer(
                      child: FractionalTranslation(
                        translation: leaveOffset,
                        child: widget.children[_previousIndex],
                      ),
                    ),
                  ),
                  Positioned.fill(
                    child: FractionalTranslation(
                      translation: enterOffset,
                      child: widget.children[_currentIndex],
                    ),
                  ),
                ],
              );
            } else {
              // 无界高度（ScrollView 内）：两页均参与 Stack 布局，
              // 容器高度取两者的最大值，避免高度不同时切换过程中截断。
              return ClipRect(
                child: Stack(
                  clipBehavior: Clip.hardEdge,
                  children: [
                    // 入场页：正常流
                    FractionalTranslation(
                      translation: enterOffset,
                      child: widget.children[_currentIndex],
                    ),
                    // 离场页：正常流（参与布局→保证容器高度 ≥ 离场页高度）
                    IgnorePointer(
                      child: FractionalTranslation(
                        translation: leaveOffset,
                        child: widget.children[_previousIndex],
                      ),
                    ),
                  ],
                ),
              );
            }
          },
        );
      },
    );
  }
}
