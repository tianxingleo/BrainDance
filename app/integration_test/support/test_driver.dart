import 'package:flutter_test/flutter_test.dart';

class TestDriver {
  const TestDriver(this.tester);

  final WidgetTester tester;

  Future<void> openLoginPage() async {
    await tester.pumpAndSettle();
  }

  Future<void> clearSessionState() async {
    await tester.pumpAndSettle();
  }

  Future<void> attachTestImage() async {
    throw UnimplementedError('待接入测试图片选择与注入逻辑。');
  }

  Future<void> attachTestVideo() async {
    throw UnimplementedError('待接入测试视频选择与注入逻辑。');
  }
}
