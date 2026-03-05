import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

Widget setTab4(BuildContext context, ScrollController scrollController) {
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
        child: Scrollbar(
          controller: scrollController,
          child: ListView.separated(
            controller: scrollController,
            itemCount: 50,
            separatorBuilder: (ctx, index) => Divider(
              height: 1,
              color: TDTheme.of(ctx).grayColor3,
            ),
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
  );
}
