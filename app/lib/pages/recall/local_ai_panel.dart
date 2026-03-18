import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../../configs/motion_tokens.dart';
import '../../widgets/bd_surfaces.dart';

class RecallLocalAiPanel extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color textColor;
  final Color darkInput;
  final bool isLocalAiPanelOpen;
  final bool isLocalModelReady;
  final bool isModelDownloading;
  final bool isLocalModelLoading;
  final bool isLocalAnswering;
  final double? modelDownloadProgress;
  final int modelDownloadedBytes;
  final int? modelDownloadTotalBytes;
  final String localAnswer;
  final String localAnswerStatus;
  final String localContextPreview;
  final String defaultModelDownloadUrl;
  final TextEditingController localModelUrlController;
  final TextEditingController localModelPathController;
  final TextEditingController localQuestionController;
  final VoidCallback onToggleOpen;
  final VoidCallback onDownloadModel;
  final VoidCallback onLoadModel;
  final VoidCallback onAskQuestion;

  const RecallLocalAiPanel({
    super.key,
    required this.theme,
    required this.isDark,
    required this.textColor,
    required this.darkInput,
    required this.isLocalAiPanelOpen,
    required this.isLocalModelReady,
    required this.isModelDownloading,
    required this.isLocalModelLoading,
    required this.isLocalAnswering,
    required this.modelDownloadProgress,
    required this.modelDownloadedBytes,
    required this.modelDownloadTotalBytes,
    required this.localAnswer,
    required this.localAnswerStatus,
    required this.localContextPreview,
    required this.defaultModelDownloadUrl,
    required this.localModelUrlController,
    required this.localModelPathController,
    required this.localQuestionController,
    required this.onToggleOpen,
    required this.onDownloadModel,
    required this.onLoadModel,
    required this.onAskQuestion,
  });

  @override
  Widget build(BuildContext context) {
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;
    final answerText = localAnswer.trim();
    final contextPreview = localContextPreview.trim();

    return Column(
      children: [
        BDPanelCard(
          padding: const EdgeInsets.all(14),
          child: InkWell(
            borderRadius: BorderRadius.circular(18),
            onTap: onToggleOpen,
            child: Row(
              children: [
                Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    color: isDark
                        ? const Color(0xFF1F2836)
                        : const Color(0xFFEAF1FB),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Icon(
                    Icons.memory_rounded,
                    color: isDark
                        ? BDDesign.colorPaperWhite
                        : BDDesign.colorInkBlack,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '本地 AI 问答',
                        style: TextStyle(
                          color: textColor,
                          fontSize: 14.5,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        isLocalModelReady ? '已就绪，点击展开端侧模型问答' : '点击展开并配置端侧模型',
                        style: TextStyle(
                          color: hintColor,
                          fontSize: 12.5,
                          height: 1.35,
                        ),
                      ),
                    ],
                  ),
                ),
                BDStatusPill(
                  label: isLocalModelReady ? 'READY' : 'OFFLINE',
                  icon: isLocalModelReady
                      ? Icons.check_circle_rounded
                      : Icons.offline_bolt_rounded,
                  color: isLocalModelReady
                      ? BDDesign.colorMutedBlue
                      : BDDesign.colorDarkRed,
                ),
                const SizedBox(width: 8),
                AnimatedRotation(
                  turns: isLocalAiPanelOpen ? 0.5 : 0,
                  duration: const Duration(milliseconds: 220),
                  child: Icon(Icons.expand_more_rounded, color: hintColor),
                ),
              ],
            ),
          ),
        ),
        AnimatedCrossFade(
          firstChild: const SizedBox.shrink(),
          secondChild: Padding(
            padding: const EdgeInsets.only(top: 10),
            child: _RecallLocalQnaPanel(
              theme: theme,
              isDark: isDark,
              textColor: textColor,
              darkInput: darkInput,
              isLocalModelReady: isLocalModelReady,
              isModelDownloading: isModelDownloading,
              isLocalModelLoading: isLocalModelLoading,
              isLocalAnswering: isLocalAnswering,
              modelDownloadProgress: modelDownloadProgress,
              modelDownloadedBytes: modelDownloadedBytes,
              modelDownloadTotalBytes: modelDownloadTotalBytes,
              localAnswerStatus: localAnswerStatus,
              answerText: answerText,
              contextPreview: contextPreview,
              defaultModelDownloadUrl: defaultModelDownloadUrl,
              localModelUrlController: localModelUrlController,
              localModelPathController: localModelPathController,
              localQuestionController: localQuestionController,
              onDownloadModel: onDownloadModel,
              onLoadModel: onLoadModel,
              onAskQuestion: onAskQuestion,
            ),
          ),
          crossFadeState: isLocalAiPanelOpen
              ? CrossFadeState.showSecond
              : CrossFadeState.showFirst,
          duration: const Duration(milliseconds: 220),
        ),
      ],
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
  final bool isLocalAnswering;
  final double? modelDownloadProgress;
  final int modelDownloadedBytes;
  final int? modelDownloadTotalBytes;
  final String localAnswerStatus;
  final String answerText;
  final String contextPreview;
  final String defaultModelDownloadUrl;
  final TextEditingController localModelUrlController;
  final TextEditingController localModelPathController;
  final TextEditingController localQuestionController;
  final VoidCallback onDownloadModel;
  final VoidCallback onLoadModel;
  final VoidCallback onAskQuestion;

  const _RecallLocalQnaPanel({
    required this.theme,
    required this.isDark,
    required this.textColor,
    required this.darkInput,
    required this.isLocalModelReady,
    required this.isModelDownloading,
    required this.isLocalModelLoading,
    required this.isLocalAnswering,
    required this.modelDownloadProgress,
    required this.modelDownloadedBytes,
    required this.modelDownloadTotalBytes,
    required this.localAnswerStatus,
    required this.answerText,
    required this.contextPreview,
    required this.defaultModelDownloadUrl,
    required this.localModelUrlController,
    required this.localModelPathController,
    required this.localQuestionController,
    required this.onDownloadModel,
    required this.onLoadModel,
    required this.onAskQuestion,
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
            '把手机上的 Qwen3-1.7B GGUF 路径填进来，当前问题会先走本地 RAG 检索，再把记忆片段喂给端侧模型。',
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
          const SizedBox(height: 14),
          TextField(
            controller: localQuestionController,
            style: TextStyle(color: textColor, fontSize: 14),
            minLines: 2,
            maxLines: 4,
            decoration: InputDecoration(
              labelText: '问一个和记忆相关的问题',
              hintText: '例如：我上次重建的那个客厅场景里有什么物体？',
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
              onPressed: (isLocalAnswering || !isLocalModelReady)
                  ? null
                  : onAskQuestion,
              icon: const Icon(Icons.auto_awesome_rounded),
              label: Text(isLocalAnswering ? '回答中...' : '开始端侧问答'),
              style: ElevatedButton.styleFrom(
                minimumSize: const Size.fromHeight(46),
                backgroundColor: isDark
                    ? const Color(0xFF1F2836)
                    : const Color(0xFF111827),
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
            ),
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
