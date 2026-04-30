/// Smoke tests: 确认配置类能正常构造，不崩溃。
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';

void main() {
  // ─── AppConfig ─────────────────────────────────────────────────────
  group('AppConfig 静态常量', () {
    test('appName 存在且等于 BrainDance', () {
      expect(AppConfig.appName, 'BrainDance');
    });

    test('version 符合 x.y.z 格式', () {
      expect(AppConfig.version, matches(r'^\d+\.\d+\.\d+$'));
    });

    test('fontFamily 存在', () {
      expect(AppConfig.fontFamily, 'HarmonyOS_Sans');
    });

    test('primaryColor alpha > 0（非完全透明）', () {
      expect(AppConfig.primaryColor.alpha, greaterThan(0));
    });

    test('accentColor alpha > 0', () {
      expect(AppConfig.accentColor.alpha, greaterThan(0));
    });
  });

  // ─── textLocalize ──────────────────────────────────────────────────
  group('textLocalize', () {
    setUp(() {
      AppConfig.langMap = {
        'test_key': 'Test Value',
        'hello': 'Hello World',
      };
    });

    test('存在 key 时返回翻译值', () {
      expect(textLocalize('test_key'), 'Test Value');
    });

    test('不存在 key 时回退到 key 本身', () {
      expect(textLocalize('nonexistent_key'), 'nonexistent_key');
    });

    test('langMap 为空时 key 回退', () {
      AppConfig.langMap = <String, String>{};
      expect(textLocalize('any_key'), 'any_key');
    });
  });

  // ─── BDMotion ──────────────────────────────────────────────────────
  group('BDMotion 动效令牌', () {
    test('三个持续时间均大于 0ms', () {
      expect(BDMotion.durationFast.inMilliseconds, greaterThan(0));
      expect(BDMotion.durationNormal.inMilliseconds, greaterThan(0));
      expect(BDMotion.durationSlow.inMilliseconds, greaterThan(0));
    });

    test('durationFast < durationNormal < durationSlow', () {
      expect(
        BDMotion.durationFast.inMilliseconds,
        lessThan(BDMotion.durationNormal.inMilliseconds),
      );
      expect(
        BDMotion.durationNormal.inMilliseconds,
        lessThan(BDMotion.durationSlow.inMilliseconds),
      );
    });

    test('动画曲线均非空', () {
      // ignore: unnecessary_type_check
      expect(BDMotion.curveEnter is Curve, isTrue);
      // ignore: unnecessary_type_check
      expect(BDMotion.curveExit is Curve, isTrue);
      // ignore: unnecessary_type_check
      expect(BDMotion.curveFluid is Curve, isTrue);
      // ignore: unnecessary_type_check
      expect(BDMotion.curveBreathe is Curve, isTrue);
    });
  });

  // ─── BDDesign ──────────────────────────────────────────────────────
  group('BDDesign 设计令牌', () {
    test('颜色常量 alpha == 255（完全不透明）', () {
      expect(BDDesign.colorPaperWhite.alpha, 255);
      expect(BDDesign.colorAshGray.alpha, 255);
      expect(BDDesign.colorInkBlack.alpha, 255);
      expect(BDDesign.colorMutedBlue.alpha, 255);
      expect(BDDesign.colorDarkRed.alpha, 255);
      expect(BDDesign.colorFadedOlive.alpha, 255);
    });

    test('圆角常量构造成功', () {
      expect(BDDesign.radiusSmall, isA<BorderRadius>());
      expect(BDDesign.radiusNormal, isA<BorderRadius>());
      expect(BDDesign.radiusLarge, isA<BorderRadius>());
    });

    test('阴影常量构造成功', () {
      expect(BDDesign.shadowLight, isA<BoxShadow>());
      expect(BDDesign.shadowElevated, isA<BoxShadow>());
    });

    test('shadowLight 模糊半径 < shadowElevated 模糊半径', () {
      expect(
        BDDesign.shadowLight.blurRadius,
        lessThan(BDDesign.shadowElevated.blurRadius),
      );
    });
  });
}
