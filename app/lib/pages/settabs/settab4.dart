import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

Widget setTab4(ScrollController scrollController) {
  return Padding(
    padding: const EdgeInsets.all(16.0),
    child: Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Scrollbar(
          controller: scrollController,
          child: ListView.separated(
            controller: scrollController,
            itemCount: 50,
            separatorBuilder: (context, index) => const Divider(height: 1, color: Color(0xFFF3F3F3)),
            itemBuilder: (context, index) => ListTile(
              title: Text('Item $index', style: const TextStyle(fontSize: 16)),
              trailing: const Icon(Icons.chevron_right, color: Colors.grey, size: 20),
              onTap: () {},
            ),
          ),
        ),
      ),
    ),
  );
}
