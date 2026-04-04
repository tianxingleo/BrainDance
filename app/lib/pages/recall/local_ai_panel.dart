import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:markdown/markdown.dart' as md;
import 'package:flutter_highlight/flutter_highlight.dart';
import 'package:flutter_highlight/themes/atom-one-dark.dart';
import 'package:flutter_highlight/themes/atom-one-light.dart';

import '../../configs/motion_tokens.dart';
import '../../services/local_model_catalog_service.dart';
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
  final String localReasoning;
  final String localAnswerStatus;
  final String localContextPreview;
  final String defaultModelDownloadUrl;
  final List<LocalModelCatalogItem> localModelCatalog;
  final String? selectedLocalModelUrl;
  final String? activeLocalModelUrl;
  final Set<String> downloadedLocalModelUrls;
  final TextEditingController localModelUrlController;
  final TextEditingController localModelPathController;
  final ValueChanged<String?> onSelectCatalogModel;
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
    required this.localReasoning,
    required this.localAnswerStatus,
    required this.localContextPreview,
    required this.defaultModelDownloadUrl,
    required this.localModelCatalog,
    required this.selectedLocalModelUrl,
    required this.activeLocalModelUrl,
    required this.downloadedLocalModelUrls,
    required this.localModelUrlController,
    required this.localModelPathController,
    required this.onSelectCatalogModel,
    required this.onDownloadModel,
    required this.onLoadModel,
  });

  @override
  Widget build(BuildContext context) {
    final answerText = localAnswer.trim();
    final reasoningText = localReasoning.trim();
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
      reasoningText: reasoningText,
      contextPreview: contextPreview,
      defaultModelDownloadUrl: defaultModelDownloadUrl,
      localModelCatalog: localModelCatalog,
      selectedLocalModelUrl: selectedLocalModelUrl,
      activeLocalModelUrl: activeLocalModelUrl,
      downloadedLocalModelUrls: downloadedLocalModelUrls,
      localModelUrlController: localModelUrlController,
      localModelPathController: localModelPathController,
      onSelectCatalogModel: onSelectCatalogModel,
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
  final String reasoningText;
  final String contextPreview;
  final String defaultModelDownloadUrl;
  final List<LocalModelCatalogItem> localModelCatalog;
  final String? selectedLocalModelUrl;
  final String? activeLocalModelUrl;
  final Set<String> downloadedLocalModelUrls;
  final TextEditingController localModelUrlController;
  final TextEditingController localModelPathController;
  final ValueChanged<String?> onSelectCatalogModel;
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
    required this.reasoningText,
    required this.contextPreview,
    required this.defaultModelDownloadUrl,
    required this.localModelCatalog,
    required this.selectedLocalModelUrl,
    required this.activeLocalModelUrl,
    required this.downloadedLocalModelUrls,
    required this.localModelUrlController,
    required this.localModelPathController,
    required this.onSelectCatalogModel,
    required this.onDownloadModel,
    required this.onLoadModel,
  });

  @override
  Widget build(BuildContext context) {
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;
    LocalModelCatalogItem? selectedCatalogModel;
    for (final item in localModelCatalog) {
      if (item.downloadUrl == selectedLocalModelUrl) {
        selectedCatalogModel = item;
        break;
      }
    }

    Widget buildModelTile({
      required BuildContext context,
      required LocalModelCatalogItem item,
      required bool isSelected,
      required VoidCallback onTap,
    }) {
      final labels = <String>[
        if (item.isRecommended) '推荐',
        if (downloadedLocalModelUrls.contains(item.downloadUrl)) '已下载',
        if (activeLocalModelUrl == item.downloadUrl) '当前使用',
      ];
      final suffix = labels.isEmpty ? '' : labels.join(' · ');
      final modelSize = item.sizeBytes != null
          ? '${(item.sizeBytes! / 1024 / 1024 / 1024).toStringAsFixed(2)} GB'
          : '';

      return Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: InkWell(
          borderRadius: BorderRadius.circular(18),
          onTap: onTap,
          child: AnimatedContainer(
            duration: BDMotion.durationFast,
            curve: Curves.easeOutCubic,
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: isSelected
                  ? BDDesign.colorMutedBlue.withValues(
                      alpha: isDark ? 0.22 : 0.10,
                    )
                  : (isDark ? darkInput : const Color(0xFFF6F8FC)),
              borderRadius: BorderRadius.circular(18),
              border: Border.all(
                color: isSelected
                    ? BDDesign.colorMutedBlue
                    : (isDark
                          ? Colors.white.withValues(alpha: 0.08)
                          : BDDesign.colorMutedBlue.withValues(alpha: 0.14)),
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    color: isSelected
                        ? BDDesign.colorMutedBlue.withValues(alpha: 0.18)
                        : (isDark
                              ? Colors.white.withValues(alpha: 0.05)
                              : Colors.white),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Icon(Icons.memory_rounded, color: textColor, size: 20),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        item.name,
                        style: TextStyle(
                          color: textColor,
                          fontSize: 14,
                          fontWeight: isSelected
                              ? FontWeight.w600
                              : FontWeight.w500,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      if (suffix.isNotEmpty || modelSize.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(top: 2),
                          child: Text(
                            [
                              if (suffix.isNotEmpty) suffix,
                              if (modelSize.isNotEmpty) modelSize,
                            ].join(' | '),
                            style: TextStyle(color: hintColor, fontSize: 12),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                    ],
                  ),
                ),
                if (isSelected) ...[
                  const SizedBox(width: 12),
                  Icon(
                    Icons.check_circle_rounded,
                    color: BDDesign.colorMutedBlue,
                    size: 22,
                  ),
                ],
              ],
            ),
          ),
        ),
      );
    }

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
          if (localModelCatalog.isNotEmpty) ...[
            const SizedBox(height: 12),
            if (selectedCatalogModel != null)
              buildModelTile(
                context: context,
                item: selectedCatalogModel,
                isSelected: false,
                onTap: () {
                  showModalBottomSheet(
                    context: context,
                    backgroundColor: isDark
                        ? const Color(0xFF1C2331)
                        : Colors.white,
                    isScrollControlled: true,
                    shape: const RoundedRectangleBorder(
                      borderRadius: BorderRadius.vertical(
                        top: Radius.circular(24),
                      ),
                    ),
                    builder: (BuildContext ctx) {
                      return SafeArea(
                        child: Padding(
                          padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Container(
                                width: 36,
                                height: 4,
                                margin: const EdgeInsets.only(bottom: 16),
                                decoration: BoxDecoration(
                                  color: isDark
                                      ? Colors.white24
                                      : Colors.black12,
                                  borderRadius: BorderRadius.circular(2),
                                ),
                              ),
                              Row(
                                children: [
                                  Expanded(
                                    child: Text(
                                      '选择端侧模型',
                                      style: TextStyle(
                                        fontSize: 18,
                                        fontWeight: FontWeight.w600,
                                        color: textColor,
                                      ),
                                    ),
                                  ),
                                  IconButton(
                                    icon: Icon(Icons.close, color: hintColor),
                                    onPressed: () => Navigator.pop(ctx),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 16),
                              ConstrainedBox(
                                constraints: BoxConstraints(
                                  maxHeight:
                                      MediaQuery.of(context).size.height * 0.5,
                                ),
                                child: ListView.builder(
                                  shrinkWrap: true,
                                  itemCount: localModelCatalog.length,
                                  itemBuilder: (context, index) {
                                    final item = localModelCatalog[index];
                                    final isSelected =
                                        item.downloadUrl ==
                                        selectedLocalModelUrl;
                                    return buildModelTile(
                                      context: ctx,
                                      item: item,
                                      isSelected: isSelected,
                                      onTap: () {
                                        Navigator.pop(ctx);
                                        onSelectCatalogModel(item.downloadUrl);
                                      },
                                    );
                                  },
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  );
                },
              ),
          ],
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
          if (reasoningText.isNotEmpty) ...[
            const SizedBox(height: 14),
            Theme(
              data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
              child: ExpansionTile(
                tilePadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 0),
                minTileHeight: 0,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                collapsedShape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                backgroundColor: isDark ? darkInput : theme.grayColor3,
                collapsedBackgroundColor: isDark ? darkInput : theme.grayColor3,
                iconColor: hintColor,
                collapsedIconColor: hintColor,
                title: Row(
                  children: [
                    Icon(
                      localAnswerStatus.contains('问答完成') || localAnswerStatus.contains('回答完成')
                          ? Icons.psychology_rounded
                          : Icons.stream_rounded,
                      size: 16,
                      color: BDDesign.colorMutedBlue,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      localAnswerStatus.contains('问答完成') || localAnswerStatus.contains('回答完成')
                          ? '查看思考过程'
                          : 'Agent 正在思考...',
                      style: TextStyle(
                        color: textColor,
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
                children: [
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
                    child: Text(
                      reasoningText,
                      style: TextStyle(
                        color: hintColor,
                        fontSize: 12.5,
                        height: 1.45,
                        fontStyle: FontStyle.italic,
                      ),
                    ),
                  ),
                ],
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
            child: answerText.isEmpty
                ? Text(
                    '模型回答会在这里流式出现。',
                    style: TextStyle(
                      color: hintColor,
                      fontSize: 13.5,
                      height: 1.5,
                    ),
                  )
                : MarkdownBody(
                    data: _prepareStreamingMarkdown(answerText),
                    builders: {
                      'code': _CodeElementBuilder(isDark: isDark),
                    },
                    styleSheet: MarkdownStyleSheet(
                      p: TextStyle(color: textColor, fontSize: 13.5, height: 1.5),
                      h1: TextStyle(color: textColor, fontSize: 18, height: 1.3, fontWeight: FontWeight.w700),
                      h2: TextStyle(color: textColor, fontSize: 16, height: 1.3, fontWeight: FontWeight.w700),
                      blockquote: TextStyle(color: hintColor, fontSize: 13, height: 1.5),
                      code: TextStyle(color: textColor, fontSize: 12.5, fontFamily: 'monospace'),
                      codeblockDecoration: BoxDecoration(
                        color: isDark ? const Color(0xFF131813) : const Color(0xFFF1F4EA),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: BDDesign.colorFadedOlive.withValues(alpha: isDark ? 0.28 : 0.18),
                        ),
                      ),
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

  String _prepareStreamingMarkdown(String text) {
    if (text.isEmpty) return text;
    final parts = text.split('```');
    if (parts.length % 2 == 0) {
      return '$text\n```';
    }
    return text;
  }
}

class _CodeElementBuilder extends MarkdownElementBuilder {
  final bool isDark;

  _CodeElementBuilder({required this.isDark});

  @override
  Widget? visitElementAfter(md.Element element, TextStyle? preferredStyle) {
    if (element.tag != 'code') return null;

    final language = element.attributes['class']?.replaceFirst('language-', '') ?? '';
    final codeText = element.textContent;

    // Check if it's an inline code block (without newlines typically, or parent isn't pre)
    // In flutter_markdown, block code has empty language if not specified, but we can't easily detect inline vs block here if not for 'class' attribute.
    // Usually inline code doesn't have the class attribute. Let's assume class exists == block or it has newlines.
    if (!codeText.contains('\n') && language.isEmpty) {
      return null; // fallback to default style sheet rendering for inline code
    }

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF12161A) : const Color(0xFFF1F4EA),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: BDDesign.colorFadedOlive.withValues(alpha: isDark ? 0.28 : 0.18),
        ),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: HighlightView(
          codeText,
          language: language.isEmpty ? 'text' : language,
          theme: isDark ? atomOneDarkTheme : atomOneLightTheme,
          padding: const EdgeInsets.all(12),
          textStyle: const TextStyle(
            fontFamily: 'monospace',
            fontSize: 12.5,
            height: 1.45,
          ),
        ),
      ),
    );
  }
}
