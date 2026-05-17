import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';

Widget setTab4(BuildContext context, ScrollController scrollController) {
  return Padding(
    padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
    child: ListView(
      children: [
        Text(
          textLocalize('set_tab4_title'),
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
          textLocalize('set_tab4_desc'),
          style: TextStyle(
            fontSize: 13,
            height: 1.45,
            color: Theme.of(context).brightness == Brightness.dark
                ? Colors.white.withValues(alpha: 0.62)
                : BDDesign.colorMutedBlue,
          ),
        ),
        const SizedBox(height: 14),
        SizedBox(
          height: 420,
          child: BDPanelCard(
            glass: true,
            child: ClipRRect(
              borderRadius: BDDesign.radiusLarge,
              child: Scrollbar(
                controller: scrollController,
                child: ListView.separated(
                  controller: scrollController,
                  itemCount: 50,
                  separatorBuilder: (ctx, index) =>
                      Divider(height: 1, color: TDTheme.of(ctx).grayColor3),
                  itemBuilder: (ctx, index) => ListTile(
                    title: Text(
                      'Item $index',
                      style: TextStyle(
                        fontSize: 16,
                        color: TDTheme.of(ctx).fontGyColor1,
                      ),
                    ),
                    trailing: Icon(
                      Icons.chevron_right,
                      color: TDTheme.of(ctx).fontGyColor3,
                      size: 20,
                    ),
                    onTap: () {},
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    ),
  );
}
