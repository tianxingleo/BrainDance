import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';

Widget setTab3(BuildContext context) {
  return Padding(
    padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
    child: ListView(
      children: [
        Text(
          '关于与存储',
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
          '保留版本信息、本地缓存和清理入口，用更像档案页的方式展示。',
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
                      textLocalize("tip_cache"),
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
