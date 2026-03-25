import 'package:braindance/floating_nav_bar.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('floating nav shows selected label and handles tap', (
    WidgetTester tester,
  ) async {
    var selectedIndex = 0;

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: Stack(
            children: [
              const Positioned.fill(
                child: BDPageBackdrop(child: SizedBox.expand()),
              ),
              StatefulBuilder(
                builder: (context, setState) {
                  return FloatingNavBar(
                    currentIndex: selectedIndex,
                    onTap: (index) {
                      setState(() {
                        selectedIndex = index;
                      });
                    },
                    items: [
                      NavIslandItem(
                        icon: Icons.history_edu_rounded,
                        label: 'Recall',
                      ),
                      NavIslandItem(
                        icon: Icons.add_rounded,
                        label: 'Create',
                        isLarge: true,
                      ),
                      NavIslandItem(
                        icon: Icons.camera_alt_rounded,
                        label: 'Record',
                      ),
                    ],
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );

    expect(
      find.byKey(const ValueKey('floating-nav-selected-content-Recall')),
      findsOneWidget,
    );
    expect(
      find.byKey(const ValueKey('floating-nav-selected-content-Record')),
      findsNothing,
    );

    await tester.tap(find.byIcon(Icons.camera_alt_rounded).hitTestable());
    await tester.pumpAndSettle();

    expect(
      find.byKey(const ValueKey('floating-nav-selected-content-Record')),
      findsOneWidget,
    );
    expect(selectedIndex, 2);
  });

  testWidgets('floating nav pill is centered with selected content', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: Stack(
            children: [
              const Positioned.fill(
                child: BDPageBackdrop(child: SizedBox.expand()),
              ),
              FloatingNavBar(
                currentIndex: 0,
                onTap: (_) {},
                items: [
                  NavIslandItem(
                    icon: Icons.history_edu_rounded,
                    label: 'Recall',
                  ),
                  NavIslandItem(
                    icon: Icons.add_rounded,
                    label: 'Create',
                    isLarge: true,
                  ),
                  NavIslandItem(
                    icon: Icons.camera_alt_rounded,
                    label: 'Record',
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );

    await tester.pumpAndSettle();

    final pillCenter = tester.getCenter(
      find.byKey(const ValueKey('floating-nav-pill')),
    );
    final contentCenter = tester.getCenter(
      find.byKey(const ValueKey('floating-nav-selected-content-Recall')),
    );

    expect((pillCenter.dx - contentCenter.dx).abs(), lessThanOrEqualTo(2.0));
  });
}
