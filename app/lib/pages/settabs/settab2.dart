import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';

Widget setTab2(VoidCallback onUpdate, BuildContext context) {
  return Padding(
    padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
    child: ListView(
      children: [
        Text(
          textLocalize('set_tab2_title'),
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: Theme.of(context).brightness == Brightness.dark
                ? BDDesign.colorPaperWhite
                : BDDesign.colorInkBlack,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          textLocalize('set_tab2_desc'),
          style: TextStyle(
            fontSize: 13,
            height: 1.45,
            color: Theme.of(context).brightness == Brightness.dark
                ? Colors.white.withValues(alpha: 0.62)
                : BDDesign.colorMutedBlue,
          ),
        ),
        const SizedBox(height: 14),
        BDPanelCard(
          glass: true,
          child: ClipRRect(
            borderRadius: BDDesign.radiusLarge,
            child: TDCellGroup(
              cells: [
                Picker.buildPicker(
                  context,
                  onUpdate: onUpdate,
                  pickerTitle: 'PickerTest1',
                  pickerIndex: 0,
                ),
                Picker.buildPicker(
                  context,
                  onUpdate: onUpdate,
                  pickerTitle: 'PickerTest2',
                  pickerIndex: 1,
                ),
              ],
            ),
          ),
        ),
      ],
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
