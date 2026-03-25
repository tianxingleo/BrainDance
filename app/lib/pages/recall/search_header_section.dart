import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/services/local_model_catalog_service.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import 'local_ai_panel.dart';
import 'search_mode.dart';

class RecallSearchHeaderSection extends StatelessWidget {
  final TDThemeData theme;
  final bool isDark;
  final Color textColor;
  final Color darkInput;
  final TextEditingController searchController;
  final RecallSearchMode searchMode;
  final String Function(RecallSearchMode mode) searchModeTitleBuilder;
  final String Function(RecallSearchMode mode) searchModeSubtitleBuilder;
  final String searchFieldHint;
  final Future<void> Function(String value) onSubmit;
  final ValueChanged<String> onChanged;
  final VoidCallback onClear;
  final VoidCallback onTapSearchMode;
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
  final List<LocalModelCatalogItem> localModelCatalog;
  final String? selectedLocalModelUrl;
  final String? activeLocalModelUrl;
  final Set<String> downloadedLocalModelUrls;
  final TextEditingController localModelUrlController;
  final TextEditingController localModelPathController;
  final ValueChanged<String?> onSelectCatalogModel;
  final Future<void> Function() onDownloadModel;
  final Future<void> Function() onLoadModel;

  const RecallSearchHeaderSection({
    super.key,
    required this.theme,
    required this.isDark,
    required this.textColor,
    required this.darkInput,
    required this.searchController,
    required this.searchMode,
    required this.searchModeTitleBuilder,
    required this.searchModeSubtitleBuilder,
    required this.searchFieldHint,
    required this.onSubmit,
    required this.onChanged,
    required this.onClear,
    required this.onTapSearchMode,
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
    return Column(
      children: [
        BDPanelCard(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          child: TextField(
            controller: searchController,
            style: TextStyle(color: textColor, fontSize: 15),
            decoration: InputDecoration(
              hintText: searchFieldHint,
              hintStyle: TextStyle(
                color: isDark
                    ? Colors.white.withValues(alpha: 0.45)
                    : BDDesign.colorMutedBlue.withValues(alpha: 0.78),
                fontSize: 15,
              ),
              prefixIcon: Icon(
                Icons.search_rounded,
                color: isDark
                    ? Colors.white.withValues(alpha: 0.5)
                    : BDDesign.colorMutedBlue,
              ),
              suffixIcon: ValueListenableBuilder<TextEditingValue>(
                valueListenable: searchController,
                builder: (context, value, _) {
                  final hasText = value.text.trim().isNotEmpty;
                  return Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      RecallSearchModeButton(
                        isDark: isDark,
                        title: searchModeTitleBuilder(searchMode),
                        icon: switch (searchMode) {
                          RecallSearchMode.cloud => Icons.cloud_rounded,
                          RecallSearchMode.local => Icons.privacy_tip_rounded,
                          RecallSearchMode.localAi =>
                            Icons.auto_awesome_rounded,
                        },
                        onTap: onTapSearchMode,
                      ),
                      if (hasText)
                        IconButton(
                          onPressed: onClear,
                          icon: Icon(
                            Icons.close_rounded,
                            color: isDark
                                ? Colors.white.withValues(alpha: 0.5)
                                : BDDesign.colorMutedBlue,
                          ),
                        ),
                    ],
                  );
                },
              ),
              filled: true,
              fillColor: Colors.transparent,
              contentPadding: const EdgeInsets.symmetric(
                vertical: 14,
                horizontal: 16,
              ),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
                borderSide: BorderSide.none,
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
                borderSide: BorderSide.none,
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
                borderSide: const BorderSide(
                  color: BDDesign.colorMutedBlue,
                  width: 1.5,
                ),
              ),
            ),
            onSubmitted: (value) {
              onSubmit(value);
            },
            onChanged: onChanged,
          ),
        ),
        if (searchMode == RecallSearchMode.localAi) ...[
          const SizedBox(height: 10),
          RecallLocalAiPanel(
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
            localAnswer: localAnswer,
            localAnswerStatus: localAnswerStatus,
            localContextPreview: localContextPreview,
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
          ),
        ],
      ],
    );
  }
}

class RecallSearchModeButton extends StatelessWidget {
  final bool isDark;
  final String title;
  final IconData icon;
  final VoidCallback onTap;

  const RecallSearchModeButton({
    super.key,
    required this.isDark,
    required this.title,
    required this.icon,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final foreground = isDark
        ? Colors.white.withValues(alpha: 0.82)
        : BDDesign.colorInkBlack;

    return Tooltip(
      message: textLocalize('recall_search_mode'),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Container(
            constraints: const BoxConstraints(minWidth: 68),
            height: 32,
            padding: const EdgeInsets.symmetric(horizontal: 10),
            decoration: BoxDecoration(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.06)
                  : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(
                color: isDark
                    ? Colors.white.withValues(alpha: 0.08)
                    : BDDesign.colorMutedBlue.withValues(alpha: 0.18),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 16, color: foreground),
                const SizedBox(width: 6),
                Flexible(
                  child: Text(
                    title,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: foreground,
                      fontSize: 11.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                const SizedBox(width: 2),
                Icon(Icons.expand_more_rounded, size: 16, color: foreground),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class RecallSearchModeSheet extends StatelessWidget {
  final RecallSearchMode selectedMode;
  final String Function(RecallSearchMode mode) titleBuilder;
  final String Function(RecallSearchMode mode) subtitleBuilder;
  final ValueChanged<RecallSearchMode> onSelect;
  final Color darkInput;

  const RecallSearchModeSheet({
    super.key,
    required this.selectedMode,
    required this.titleBuilder,
    required this.subtitleBuilder,
    required this.onSelect,
    required this.darkInput,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = AppConfig.isNightMode;
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;

    Widget modeTile({required RecallSearchMode mode, required IconData icon}) {
      final selected = selectedMode == mode;
      return InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: () => onSelect(mode),
        child: AnimatedContainer(
          duration: BDMotion.durationFast,
          curve: Curves.easeOutCubic,
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: selected
                ? BDDesign.colorMutedBlue.withValues(
                    alpha: isDark ? 0.22 : 0.10,
                  )
                : (isDark ? darkInput : const Color(0xFFF6F8FC)),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: selected
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
                  color: selected
                      ? BDDesign.colorMutedBlue.withValues(alpha: 0.18)
                      : (isDark
                            ? Colors.white.withValues(alpha: 0.05)
                            : Colors.white),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: textColor, size: 20),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      titleBuilder(mode),
                      style: TextStyle(
                        color: textColor,
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      subtitleBuilder(mode),
                      style: TextStyle(
                        color: hintColor,
                        fontSize: 12.5,
                        height: 1.35,
                      ),
                    ),
                  ],
                ),
              ),
              if (selected)
                const Icon(
                  Icons.check_circle_rounded,
                  color: BDDesign.colorMutedBlue,
                ),
            ],
          ),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
      child: BDPanelCard(
        padding: const EdgeInsets.fromLTRB(18, 18, 18, 12),
        child: SafeArea(
          top: false,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                textLocalize('recall_search_mode'),
                style: TextStyle(
                  color: textColor,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                '选择当前搜索栏要优先使用的检索方式。',
                style: TextStyle(
                  color: hintColor,
                  fontSize: 12.5,
                  height: 1.35,
                ),
              ),
              const SizedBox(height: 16),
              modeTile(mode: RecallSearchMode.cloud, icon: Icons.cloud_rounded),
              const SizedBox(height: 10),
              modeTile(
                mode: RecallSearchMode.local,
                icon: Icons.privacy_tip_rounded,
              ),
              const SizedBox(height: 10),
              modeTile(
                mode: RecallSearchMode.localAi,
                icon: Icons.auto_awesome_rounded,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
