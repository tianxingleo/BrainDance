import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

/// 与 ai_engine/3dgs/src/core/worker.py 中 _run_video_dual_chain 写入的
/// 日志关键词保持一致。修改这些字符串前必须同步更新 worker.py。
class DualChainLogKeywords {
  static const String fastDone = '⚡ 快链完成';
  static const String fastFailed = '⚠️ 快链失败';
  static const String slowDone = '🐢 慢链完成';
  static const String slowFailed = '⚠️ 慢链失败';
}

/// 双链里程碑：用于驱动前端模型下载与缓存替换。
enum DualChainMilestone { fastReady, fastFailed, slowReady, slowFailed }

/// 跨页面广播的双链事件。Recall 页负责发，Viewer 页负责听。
class DualChainEvent {
  final String sceneId;
  final DualChainMilestone milestone;
  final String? displayName;

  const DualChainEvent({
    required this.sceneId,
    required this.milestone,
    this.displayName,
  });
}

/// 全局广播流。当 Recall 检测到新的双链里程碑时往这里 add，
/// WebGL Viewer 等独立页面订阅以做相应提示或重载。
final StreamController<DualChainEvent> _dualChainEventController =
    StreamController<DualChainEvent>.broadcast();
Stream<DualChainEvent> get dualChainEventStream =>
    _dualChainEventController.stream;
void emitDualChainEvent(DualChainEvent event) {
  if (!_dualChainEventController.isClosed) {
    _dualChainEventController.add(event);
  }
}

/// 检查任务最新一次 logs 列表，返回新出现的里程碑。
/// [knownMilestones] 是该 taskId 已经处理过的里程碑集合，调用方负责持久化。
Set<DualChainMilestone> detectNewMilestones({
  required List<String> allLogs,
  required Set<DualChainMilestone> knownMilestones,
}) {
  final found = <DualChainMilestone>{};
  for (final msg in allLogs) {
    if (msg.contains(DualChainLogKeywords.fastDone)) {
      found.add(DualChainMilestone.fastReady);
    } else if (msg.contains(DualChainLogKeywords.fastFailed)) {
      found.add(DualChainMilestone.fastFailed);
    }
    if (msg.contains(DualChainLogKeywords.slowDone)) {
      found.add(DualChainMilestone.slowReady);
    } else if (msg.contains(DualChainLogKeywords.slowFailed)) {
      found.add(DualChainMilestone.slowFailed);
    }
  }
  return found.difference(knownMilestones);
}

/// 当慢链覆盖完成时，删除 webgl_viewer 在 appDocDir 下基于 URL 派生的缓存文件，
/// 让下次打开 Viewer 时自动重新下载新版本。
///
/// 缓存命名规则与 webgl_viewer.dart:_prepareModelAndLoad 保持一致：
///   sanitizedFileName = uri.path.replaceAll('/', '_').replaceAll('\\', '_')
/// 一并清理 .tmp / .meta 残留。
Future<void> invalidateViewerCacheForUrl(String url) async {
  try {
    final uri = Uri.parse(url);
    final sanitized = uri.path.replaceAll('/', '_').replaceAll('\\', '_');
    if (sanitized.isEmpty) return;
    final dir = await getApplicationDocumentsDirectory();
    final base = '${dir.path}/$sanitized';
    for (final suffix in const ['', '.tmp', '.meta']) {
      final f = File('$base$suffix');
      if (await f.exists()) {
        await f.delete();
      }
    }
    debugPrint('[DualChainPhase] invalidated viewer cache for $sanitized');
  } catch (e) {
    debugPrint('[DualChainPhase] invalidate cache error: $e');
  }
}
