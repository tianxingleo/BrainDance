import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/supabase_config.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

Widget setTab1(WidgetRef ref) {
  return Padding(
    padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (SupabaseConfig.isAdminMode)
          TDButton(
            text: textLocalize('set_admin_enabled'),
            type: TDButtonType.outline,
            theme: TDButtonTheme.primary,
            isBlock: true,
            shape: TDButtonShape.round,
            onTap: () {},
          ),
      ],
    ),
  );
}
