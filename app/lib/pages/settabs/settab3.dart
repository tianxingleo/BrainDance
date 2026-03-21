import 'package:braindance/configs/app_config.dart';
import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:braindance/configs/motion_tokens.dart';

Widget setTab3(BuildContext context) {
  return Padding(
    padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
    child: ListView(
      children: [
        BDPanelCard(
          child: ClipRRect(
            borderRadius: BDDesign.radiusLarge,
            child: TDCellGroup(
              cells: [
                TDCell(
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
                    TDToast.showText(
                      textLocalize('tip_cache'),
                      context: context,
                    );
                    await DirSystem.deleteDir(await DirFinder.cacheDir());
                  },
                ),
              ],
            ),
          ),
        ),
      ],
    ),
  );
}
