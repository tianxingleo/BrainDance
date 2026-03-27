import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'package:braindance/main.dart' as app;

Future<void> launchBrainDanceApp(WidgetTester tester) async {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  app.main();
  await tester.pump();
  await tester.pump(const Duration(seconds: 1));
}

SupabaseClient get testSupabaseClient => Supabase.instance.client;
