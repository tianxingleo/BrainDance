/// 模型网格通用工具函数
///
/// 提供模型名称解析、颜色辅助、相似度格式化等可复用工具。

import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

/// 从模型数据中解析显示名称
///
/// 优先取 [display_name]，其次取第一个非空 [tags] 项，再次取 [scene_id]，
/// 均为空时回退到 [fallback]。
String modelDisplayName(
  Map<String, dynamic> model, {
  String fallback = 'Unknown Scene',
}) {
  final displayName = model['display_name']?.toString().trim() ?? '';
  if (displayName.isNotEmpty) {
    return displayName;
  }

  final tags = model['tags'];
  if (tags is List) {
    for (final tag in tags) {
      final value = tag?.toString().trim() ?? '';
      if (value.isNotEmpty) {
        return value;
      }
    }
  }

  final sceneId = model['scene_id']?.toString().trim() ?? '';
  if (sceneId.isNotEmpty) {
    return sceneId;
  }

  return fallback;
}

/// 根据深色/浅色模式返回主文本颜色
Color resolveTextColor(bool isDark) =>
    isDark ? const Color(0xFFFFFFFF) : const Color(0xFF333333);

/// 根据深色/浅色模式返回提示文本颜色
Color resolveHintTextColor(bool isDark, TDThemeData theme) =>
    isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3;

/// 将 0-1 的相似度值格式化为百分比字符串（保留一位小数）
String formatSimilarity(double value) =>
    '${(value * 100).toStringAsFixed(1)}%';

/// 构建带提示的相似度徽章
Widget buildSimilarityBadge({required Widget child}) {
  return Tooltip(
    message: '匹配度',
    preferBelow: false,
    verticalOffset: 12,
    child: child,
  );
}
