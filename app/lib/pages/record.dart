import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import '../app_configs.dart';

class RecordPage extends StatelessWidget {
  const RecordPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Record Page')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TDButton(text: 'Button1', onTap: () {}),
            const SizedBox(height: 20),
            TDButton(
              text: 'Button2',
              onTap: () {},
            ),
          ],
        ),
      ),
    );
  }
}
