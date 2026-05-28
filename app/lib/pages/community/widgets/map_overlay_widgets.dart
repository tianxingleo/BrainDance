import 'package:braindance/configs/motion_tokens.dart';
import 'package:flutter/material.dart';

/// 地图右上角的细提示药丸，用半透明背景 + 12px 字号说明地图操作。
class MapHintPill extends StatelessWidget {
  final String text;
  final bool isDark;
  final Color hintColor;

  const MapHintPill({
    super.key,
    required this.text,
    required this.isDark,
    required this.hintColor,
  });

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: (isDark ? Colors.black : Colors.white).withValues(alpha: 0.76),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        child: Text(
          text,
          style: TextStyle(color: hintColor, fontSize: 12),
        ),
      ),
    );
  }
}

/// 缩放控件：+ / − 两个按钮叠在一个圆角面板里。
class MapZoomControls extends StatelessWidget {
  final bool isDark;
  final VoidCallback onZoomIn;
  final VoidCallback onZoomOut;

  const MapZoomControls({
    super.key,
    required this.isDark,
    required this.onZoomIn,
    required this.onZoomOut,
  });

  @override
  Widget build(BuildContext context) {
    final bgColor =
        (isDark ? Colors.black : Colors.white).withValues(alpha: 0.82);
    final iconColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            onPressed: onZoomIn,
            icon: Icon(Icons.add_rounded, color: iconColor),
          ),
          Container(
            width: 24,
            height: 1,
            color: iconColor.withValues(alpha: 0.18),
          ),
          IconButton(
            onPressed: onZoomOut,
            icon: Icon(Icons.remove_rounded, color: iconColor),
          ),
        ],
      ),
    );
  }
}

/// 圆形定位（"使用当前位置"）按钮，与 [MapZoomControls] 同色调，方便竖排堆叠。
class MapLocateButton extends StatelessWidget {
  final bool isDark;
  final bool loading;
  final VoidCallback? onPressed;

  const MapLocateButton({
    super.key,
    required this.isDark,
    required this.loading,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    final bgColor =
        (isDark ? Colors.black : Colors.white).withValues(alpha: 0.82);
    final iconColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    return Material(
      color: bgColor,
      shape: const CircleBorder(),
      elevation: 0,
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: loading ? null : onPressed,
        child: SizedBox(
          width: 44,
          height: 44,
          child: loading
              ? Center(
                  child: SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor:
                          AlwaysStoppedAnimation(BDDesign.colorMutedBlue),
                    ),
                  ),
                )
              : Icon(Icons.my_location_rounded, color: iconColor, size: 22),
        ),
      ),
    );
  }
}

/// 屏幕中心的十字准星。叠在 FlutterMap 上方，固定不随地图平移。
/// 由 [BDDesign.colorMutedBlue] 的图标 + 同色光晕组成，确保亮/暗色背景下都能看到。
class MapCenterCrosshair extends StatelessWidget {
  const MapCenterCrosshair({super.key});

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Stack(
              alignment: Alignment.center,
              children: [
                Container(
                  width: 56,
                  height: 56,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        BDDesign.colorMutedBlue.withValues(alpha: 0.20),
                        BDDesign.colorMutedBlue.withValues(alpha: 0.0),
                      ],
                    ),
                  ),
                ),
                Icon(
                  Icons.location_on_rounded,
                  color: BDDesign.colorMutedBlue,
                  size: 44,
                  shadows: const [
                    Shadow(
                      color: Color(0x66000000),
                      blurRadius: 6,
                      offset: Offset(0, 2),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 4),
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.4),
                shape: BoxShape.circle,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
