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
                        icon: Icons.camera_alt_rounded,
                        label: 'Record',
                      ),
                      NavIslandItem(
                        icon: Icons.auto_awesome_rounded,
                        label: 'Generate',
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

    expect(find.text('Recall'), findsOneWidget);
    expect(find.text('Record'), findsNothing);

    await tester.tap(find.byIcon(Icons.camera_alt_rounded));
    await tester.pumpAndSettle();

    expect(find.text('Record'), findsOneWidget);
    expect(selectedIndex, 1);
  });
}
