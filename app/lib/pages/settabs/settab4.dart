import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/widgets/bd_surfaces.dart';

Widget setTab4(BuildContext context, ScrollController scrollController) {
  return Padding(
    padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
    child: ListView(
      children: [
        Text(
          '扩展列表',
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
          '保留滚动测试内容，但外观切到新的面板系统里，避免这里还是默认列表气质。',
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
