import 'package:flutter/material.dart';

/// 状态分类配置
class StatusCategory {
  final String status;
  final String labelKey;
  final IconData icon;
  final Color color;
  final int priority;

  const StatusCategory({
    required this.status,
    required this.labelKey,
    required this.icon,
    required this.color,
    required this.priority,
  });
}

/// 预定义的状态分类

const List<StatusCategory> statusCategories = [
  StatusCategory(
    status: 'pending',
    labelKey: 'status_pending',
    icon: Icons.schedule,
    color: Colors.orange,
    priority: 1,
  ),
  StatusCategory(
    status: 'processing',
    labelKey: 'status_processing',
    icon: Icons.sync,
    color: Colors.blue,
    priority: 2,
  ),
  StatusCategory(
    status: 'completed',
    labelKey: 'status_completed',
    icon: Icons.check_circle,
    color: Colors.green,
    priority: 3,
  ),
  StatusCategory(
    status: 'failed',
    labelKey: 'status_failed',
    icon: Icons.error,
    color: Colors.red,
    priority: 4,
  ),
];