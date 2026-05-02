import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';

Widget setTab3(BuildContext context) {
  return Padding(
    padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        BDPanelCard(
          glass: true,
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Column(
            children: [
              _ManageInfoRow(
                title: textLocalize('set_ver'),
                value: AppConfig.version,
              ),
              const _ManageDivider(),
              _ManageInfoRow(
                title: textLocalize('set_pub'),
                value: AppConfig.publishDate,
              ),
              const _ManageDivider(),
              _ManageActionRow(
                title: textLocalize('set_cache'),
                onTap: () async {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text(textLocalize('tip_cache'))),
                  );
                  await DirSystem.deleteDir(await DirFinder.cacheDir());
                },
              ),
            ],
          ),
        ),
      ],
    ),
  );
}

class _ManageInfoRow extends StatelessWidget {
  final String title;
  final String value;

  const _ManageInfoRow({required this.title, required this.value});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final titleColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final valueColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
      child: Row(
        children: [
          Expanded(
            child: Text(
              title,
              style: TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w600,
                color: titleColor,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            value,
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w500,
              color: valueColor,
            ),
          ),
        ],
      ),
    );
  }
}

class _ManageActionRow extends StatelessWidget {
  final String title;
  final Future<void> Function() onTap;

  const _ManageActionRow({required this.title, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final actionColor = isDark
        ? BDDesign.colorMutedBlueLight
        : BDDesign.colorMutedBlue;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(20),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  title,
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                    color: textColor,
                  ),
                ),
              ),
              Icon(Icons.chevron_right_rounded, color: actionColor, size: 20),
            ],
          ),
        ),
      ),
    );
  }
}

class _ManageDivider extends StatelessWidget {
  const _ManageDivider();

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Divider(
      height: 1,
      indent: 18,
      endIndent: 18,
      color: isDark
          ? Colors.white.withValues(alpha: 0.08)
          : BDDesign.colorMutedBlue.withValues(alpha: 0.10),
    );
  }
}
