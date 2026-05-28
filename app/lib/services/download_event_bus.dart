import 'dart:async';

class ModelDownloadEvent {
  final String url;
  final double progress;
  final bool isComplete;
  /// 已下载字节数
  final int downloadedBytes;
  /// 文件总字节数（-1 表示未知）
  final int totalBytes;
  /// 下载被用户取消
  final bool isCancelled;
  /// 本地文件已删除，状态应重置为未下载
  final bool isDeleted;

  ModelDownloadEvent({
    required this.url,
    required this.progress,
    this.isComplete = false,
    this.downloadedBytes = 0,
    this.totalBytes = -1,
    this.isCancelled = false,
    this.isDeleted = false,
  });
}

final downloadEventBus = StreamController<ModelDownloadEvent>.broadcast();

/// 关闭全局事件总线，释放监听器引用。应在 App 退出时调用。
void disposeDownloadEventBus() {
  downloadEventBus.close();
}
