import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

class UIRobot {
  const UIRobot(this.tester);

  final WidgetTester tester;

  Future<void> tapByText(String text) async {
    final finder = find.text(text);
    expect(finder, findsWidgets);
    await tester.tap(finder.first);
    await tester.pumpAndSettle();
  }

  Future<void> enterTextByKey(Key key, String value) async {
    final finder = find.byKey(key);
    expect(finder, findsOneWidget);
    await tester.enterText(finder, value);
    await tester.pumpAndSettle();
  }
}
