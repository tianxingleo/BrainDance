import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/app_config.dart';

Widget setTab2(VoidCallback onUpdate, BuildContext context) {
  return Padding(
    padding: const EdgeInsets.all(16.0),
    child: Container(
      decoration: BoxDecoration(
        color: TDTheme.of(context).whiteColor1,
        borderRadius: BorderRadius.circular(TDTheme.of(context).radiusLarge),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(TDTheme.of(context).radiusLarge),
        child: TDCellGroup(
          cells: [
            Picker.buildPicker(
              onUpdate: onUpdate,
              context,
              pickerTitle: 'PickerTest1',
              pickerIndex: 0,
            ),
            Picker.buildPicker(
              onUpdate: onUpdate,
              context,
              pickerTitle: 'PickerTest2',
              pickerIndex: 1,
            ),
          ],
        ),
      ),
    ),
  );
}

class Picker {
  static final List<int> pickerSelectedIndex = List.filled(2, -1);
  static TDCell buildPicker(
    BuildContext context, {
    VoidCallback? onUpdate,
    String pickerTitle = '',
    int pickerIndex = 0,
    List<String> pickerData = const ['Option 1', 'Option 2', 'Option 3'],
  }) {
    final int selectedIndex = pickerSelectedIndex[pickerIndex];
    return TDCell(
      //版本信息
      arrow: true,
      title: pickerTitle,
      note: (selectedIndex == -1)
          ? textLocalize("pick_null")
          : pickerData[selectedIndex],
      onClick: (click) {
        TDPicker.showMultiPicker(
          context,
          titleHeight: 40,
          pickerHeight: 200,
          title: pickerTitle,
          onConfirm: (selected) {
            pickerSelectedIndex[pickerIndex] = selected[0];
            Navigator.of(context).pop();
            onUpdate?.call();
          },
          data: [pickerData],
          initialIndexes: (selectedIndex == -1) ? [0] : [selectedIndex],
        );
      },
    );
  }
}
