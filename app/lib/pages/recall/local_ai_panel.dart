import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../../configs/motion_tokens.dart';
import '../../widgets/bd_surfaces.dart';

class RecallLocalAiPanel extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color textColor;
  final Color darkInput;
  final bool isLocalModelReady;
  final bool isModelDownloading;
  final bool isLocalModelLoading;
  final double? modelDownloadProgress;
  final int modelDownloadedBytes;
  final int? modelDownloadTotalBytes;
  final String localAnswer;
  final String localAnswerStatus;
  final String localContextPreview;
  final String defaultModelDownloadUrl;
  final TextEditingController localModelUrlController;
  final TextEditingController localModelPathController;
  final VoidCallback onDownloadModel;
  final VoidCallback onLoadModel;

  const RecallLocalAiPanel({
    super.key,
    required this.theme,
    required this.isDark,
    required this.textColor,
    required this.darkInput,
    required this.isLocalModelReady,
    required this.isModelDownloading,
    required this.isLocalModelLoading,
    required this.modelDownloadProgress,
    required this.modelDownloadedBytes,
    required this.modelDownloadTotalBytes,
    required this.localAnswer,
    required this.localAnswerStatus,
    required this.localContextPreview,
    required this.defaultModelDownloadUrl,
    required this.localModelUrlController,
    required this.localModelPathController,
    required this.onDownloadModel,
    required this.onLoadModel,
  });

  @override
  Widget build(BuildContext context) {
    final answerText = localAnswer.trim();
    final contextPreview = localContextPreview.trim();

    return _RecallLocalQnaPanel(
      theme: theme,
      isDark: isDark,
      textColor: textColor,
      darkInput: darkInput,
      isLocalModelReady: isLocalModelReady,
      isModelDownloading: isModelDownloading,
      isLocalModelLoading: isLocalModelLoading,
      modelDownloadProgress: modelDownloadProgress,
      modelDownloadedBytes: modelDownloadedBytes,
      modelDownloadTotalBytes: modelDownloadTotalBytes,
      localAnswerStatus: localAnswerStatus,
      answerText: answerText,
      contextPreview: contextPreview,
      defaultModelDownloadUrl: defaultModelDownloadUrl,
      localModelUrlController: localModelUrlController,
      localModelPathController: localModelPathController,
      onDownloadModel: onDownloadModel,
      onLoadModel: onLoadModel,
    );
  }
}

class _RecallLocalQnaPanel extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color textColor;
  final Color darkInput;
  final bool isLocalModelReady;
  final bool isModelDownloading;
  final bool isLocalModelLoading;
  final double? modelDownloadProgress;
  final int modelDownloadedBytes;
  final int? modelDownloadTotalBytes;
  final String localAnswerStatus;
  final String answerText;
  final String contextPreview;
  final String defaultModelDownloadUrl;
  final TextEditingController localModelUrlController;
  final TextEditingController localModelPathController;
  final VoidCallback onDownloadModel;
  final VoidCallback onLoadModel;

  const _RecallLocalQnaPanel({
    required this.theme,
    required this.isDark,
    required this.textColor,
    required this.darkInput,
    required this.isLocalModelReady,
    required this.isModelDownloading,
    required this.isLocalModelLoading,
    required this.modelDownloadProgress,
    required this.modelDownloadedBytes,
    required this.modelDownloadTotalBytes,
    required this.localAnswerStatus,
    required this.answerText,
    required this.contextPreview,
    required this.defaultModelDownloadUrl,
    required this.localModelUrlController,
    required this.localModelPathController,
    required this.onDownloadModel,
    required this.onLoadModel,
  });

  @override
  Widget build(BuildContext context) {
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;

    return BDPanelCard(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.memory_rounded,
                size: 18,
                color: isDark
                    ? BDDesign.colorPaperWhite
                    : BDDesign.colorInkBlack,
              ),
              const SizedBox(width: 8),
              Text(
                'Qwen3-1.7B 端侧问答',
                style: TextStyle(
                  color: textColor,
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const Spacer(),
              BDStatusPill(
                label: isLocalModelReady ? 'READY' : 'OFFLINE',
                icon: isLocalModelReady
                    ? Icons.check_circle_rounded
                    : Icons.offline_bolt_rounded,
                color: isLocalModelReady
                    ? BDDesign.colorMutedBlue
                    : BDDesign.colorDarkRed,
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            '切到上方的“本地 AI 问答”模式后，直接在主搜索框里提问。这里仅保留端侧模型下载、加载和回答状态。',
            style: TextStyle(color: hintColor, fontSize: 12.5, height: 1.4),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: localModelUrlController,
            style: TextStyle(color: textColor, fontSize: 14),
            minLines: 2,
            maxLines: 3,
            decoration: InputDecoration(
              labelText: '模型下载链接',
              hintText: defaultModelDownloadUrl,
              filled: true,
              fillColor: Colors.transparent,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
              ),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: isModelDownloading ? null : onDownloadModel,
              icon: isModelDownloading
                  ? SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: isDark
                            ? BDDesign.colorPaperWhite
                            : BDDesign.colorInkBlack,
                      ),
                    )
                  : const Icon(Icons.download_rounded),
              label: Text(isModelDownloading ? '下载中...' : '下载到应用私有目录'),
              style: ElevatedButton.styleFrom(
                minimumSize: const Size.fromHeight(46),
                backgroundColor: isDark
                    ? const Color(0xFF243042)
                    : const Color(0xFF24415E),
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
            ),
          ),
          if (isModelDownloading || modelDownloadProgress != null) ...[
            const SizedBox(height: 10),
            LinearProgressIndicator(
              value: modelDownloadProgress,
              minHeight: 8,
              borderRadius: BorderRadius.circular(999),
            ),
            const SizedBox(height: 6),
            Text(
              modelDownloadTotalBytes == null
                  ? '已下载：${(modelDownloadedBytes / 1024 / 1024).toStringAsFixed(1)} MB'
                  : '下载进度：${(modelDownloadProgress! * 100).toStringAsFixed(1)}% · ${(modelDownloadedBytes / 1024 / 1024).toStringAsFixed(1)} / ${(modelDownloadTotalBytes! / 1024 / 1024).toStringAsFixed(1)} MB',
              style: TextStyle(color: hintColor, fontSize: 12),
            ),
          ],
          const SizedBox(height: 12),
          TextField(
            controller: localModelPathController,
            style: TextStyle(color: textColor, fontSize: 14),
            decoration: InputDecoration(
              labelText: '模型绝对路径',
              hintText: '应用私有目录中的本地路径',
              filled: true,
              fillColor: Colors.transparent,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
              ),
            ),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: isLocalModelLoading ? null : onLoadModel,
              icon: isLocalModelLoading
                  ? SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: isDark
                            ? BDDesign.colorPaperWhite
                            : BDDesign.colorInkBlack,
                      ),
                    )
                  : const Icon(Icons.play_arrow_rounded),
              label: Text(isLocalModelLoading ? '加载中...' : '加载端侧模型'),
              style: ElevatedButton.styleFrom(
                minimumSize: const Size.fromHeight(46),
                backgroundColor: BDDesign.colorMutedBlue,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            localAnswerStatus,
            style: TextStyle(color: hintColor, fontSize: 12, height: 1.35),
          ),
          if (contextPreview.isNotEmpty) ...[
            const SizedBox(height: 14),
            Text(
              '本次喂给模型的记忆片段',
              style: TextStyle(
                color: textColor,
                fontSize: 13,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 8),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: isDark ? darkInput : theme.grayColor3,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                contextPreview,
                style: TextStyle(
                  color: hintColor,
                  fontSize: 12.5,
                  height: 1.45,
                ),
                maxLines: 10,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
          const SizedBox(height: 14),
          Text(
            '模型回答',
            style: TextStyle(
              color: textColor,
              fontSize: 13,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Container(
            width: double.infinity,
            constraints: const BoxConstraints(minHeight: 120),
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: isDark ? darkInput : theme.grayColor3,
              borderRadius: BorderRadius.circular(18),
            ),
            child: Text(
              answerText.isEmpty ? '模型回答会在这里流式出现。' : answerText,
              style: TextStyle(
                color: answerText.isEmpty ? hintColor : textColor,
                fontSize: 13.5,
                height: 1.5,
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '提示：优先点“下载到应用私有目录”，这样不需要 adb，也不用手动处理 Android/data 路径。',
            style: TextStyle(
              color: hintColor.withValues(alpha: 0.9),
              fontSize: 11.5,
              height: 1.35,
            ),
          ),
        ],
      ),
    );
  }
}
