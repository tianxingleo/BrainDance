import 'package:flutter/material.dart';

Widget setTab4(ScrollController scrollController) {
  return Scrollbar(
    controller: scrollController,
    child: ListView.builder(
      controller: scrollController,
      itemCount: 50,
      itemBuilder: (context, index) => ListTile(title: Text('Item $index')),
    ),
  );
}
