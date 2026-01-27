import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/app_config.dart';

Widget setTab3(BuildContext context) {
  return TDCellGroup(
    cells: [
      TDCell(
        //版本信息
        arrow: false,
        title: textLocalize('set_ver'),
        note: AppConfig.version,
      ),
      TDCell(
        arrow: false,
        title: textLocalize('set_pub'),
        note: AppConfig.publishDate,
      ),
      TDCell(
        arrow: false,
        title: textLocalize('set_cache'),
        onClick: (cell) async {
          TDToast.showText(textLocalize("tip_cache"), context: context);
          await DirSystem.deleteDir(await DirFinder.cacheDir());
        },
      ),
    ],
  );
}
