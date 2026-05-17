import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/services/local_model_catalog_service.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import 'local_ai_panel.dart';
import 'search_mode.dart';

class RecallSearchHeaderSection extends StatefulWidget {
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
  State<RecallSearchHeaderSection> createState() =>
      _RecallSearchHeaderSectionState();
}

class _RecallSearchHeaderSectionState extends State<RecallSearchHeaderSection> {
  static final BorderRadius _searchFieldRadius = BorderRadius.circular(16);
  late final FocusNode _searchFocusNode;
  bool _isSearchFocused = false;

  @override
  void initState() {
    super.initState();
    _searchFocusNode = FocusNode()
      ..addListener(() {
        if (!mounted) {
          return;
        }
        setState(() {
          _isSearchFocused = _searchFocusNode.hasFocus;
        });
      });
  }

  @override
  void dispose() {
    _searchFocusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final panelBackground = widget.isDark
        ? AppTheme.darkSurface.withValues(alpha: 0.94)
        : BDDesign.colorPaperWhite.withValues(alpha: 0.94);
    final panelBorderColor = _isSearchFocused
        ? BDDesign.colorMutedBlue
        : (widget.isDark
              ? Colors.white.withValues(alpha: 0.08)
              : BDDesign.colorMutedBlue.withValues(alpha: 0.10));

    return Column(
      children: [
        AnimatedContainer(
          duration: BDMotion.durationFast,
          curve: Curves.easeOutCubic,
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: panelBackground,
            borderRadius: _searchFieldRadius,
            border: Border.all(
              color: panelBorderColor,
              width: _isSearchFocused ? 1.5 : 1,
            ),
            boxShadow: [
              widget.isDark ? BDDesign.shadowLight : BDDesign.shadowElevated,
            ],
          ),
          child: TextField(
            focusNode: _searchFocusNode,
            controller: widget.searchController,
            style: TextStyle(color: widget.textColor, fontSize: 15),
            minLines: 1,
            maxLines: 5,
            textInputAction: TextInputAction.search,
            decoration: InputDecoration(
              isDense: true,
              hintText: widget.searchFieldHint,
              hintStyle: TextStyle(
                color: widget.isDark
                    ? Colors.white.withValues(alpha: 0.45)
                    : BDDesign.colorMutedBlue.withValues(alpha: 0.78),
                fontSize: 15,
              ),
              prefixIcon: Icon(
                Icons.search_rounded,
                color: widget.isDark
                    ? Colors.white.withValues(alpha: 0.5)
                    : BDDesign.colorMutedBlue,
              ),
              suffixIcon: ValueListenableBuilder<TextEditingValue>(
                valueListenable: widget.searchController,
                builder: (context, value, _) {
                  final hasText = value.text.trim().isNotEmpty;
                  return Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      RecallSearchModeButton(
                        isDark: widget.isDark,
                        icon: switch (widget.searchMode) {
                          RecallSearchMode.cloud => Icons.cloud_rounded,
                          RecallSearchMode.local => Icons.privacy_tip_rounded,
                          RecallSearchMode.localAi =>
                            Icons.auto_awesome_rounded,
                          RecallSearchMode.agent =>
                            Icons.travel_explore_rounded,
                        },
                        onTap: widget.onTapSearchMode,
                      ),
                      if (hasText)
                        IconButton(
                          onPressed: widget.onClear,
                          visualDensity: VisualDensity.compact,
                          constraints: const BoxConstraints.tightFor(
                            width: 36,
                            height: 36,
                          ),
                          padding: EdgeInsets.zero,
                          splashColor: Colors.transparent,
                          highlightColor: Colors.transparent,
                          hoverColor: Colors.transparent,
                          focusColor: Colors.transparent,
                          icon: Icon(
                            Icons.close_rounded,
                            size: 18,
                            color: widget.isDark
                                ? Colors.white.withValues(alpha: 0.5)
                                : BDDesign.colorMutedBlue,
                          ),
                        ),
                    ],
                  );
                },
              ),
              suffixIconConstraints: const BoxConstraints(
                minWidth: 0,
                minHeight: 0,
              ),
              filled: true,
              fillColor: Colors.transparent,
              contentPadding: const EdgeInsets.symmetric(
                vertical: 14,
                horizontal: 16,
              ),
              border: OutlineInputBorder(
                borderRadius: _searchFieldRadius,
                borderSide: BorderSide.none,
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: _searchFieldRadius,
                borderSide: BorderSide.none,
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: _searchFieldRadius,
                borderSide: BorderSide.none,
              ),
            ),
            onSubmitted: (value) {
              widget.onSubmit(value);
            },
            onChanged: widget.onChanged,
          ),
        ),
        if (widget.searchMode == RecallSearchMode.localAi) ...[
          const SizedBox(height: 10),
          RecallLocalAiPanel(
            theme: widget.theme,
            isDark: widget.isDark,
            textColor: widget.textColor,
            darkInput: widget.darkInput,
            isLocalModelReady: widget.isLocalModelReady,
            isModelDownloading: widget.isModelDownloading,
            isLocalModelLoading: widget.isLocalModelLoading,
            modelDownloadProgress: widget.modelDownloadProgress,
            modelDownloadedBytes: widget.modelDownloadedBytes,
            modelDownloadTotalBytes: widget.modelDownloadTotalBytes,
            localAnswer: widget.localAnswer,
            localReasoning: widget.localReasoning,
            localAnswerStatus: widget.localAnswerStatus,
            localContextPreview: widget.localContextPreview,
            defaultModelDownloadUrl: widget.defaultModelDownloadUrl,
            localModelCatalog: widget.localModelCatalog,
            selectedLocalModelUrl: widget.selectedLocalModelUrl,
            activeLocalModelUrl: widget.activeLocalModelUrl,
            downloadedLocalModelUrls: widget.downloadedLocalModelUrls,
            localModelUrlController: widget.localModelUrlController,
            localModelPathController: widget.localModelPathController,
            onSelectCatalogModel: widget.onSelectCatalogModel,
            onDownloadModel: widget.onDownloadModel,
            onLoadModel: widget.onLoadModel,
          ),
        ] else if (widget.searchMode == RecallSearchMode.agent) ...[
          const SizedBox(height: 10),
          BDPanelCard(
            padding: const EdgeInsets.all(16),
            child: Text(
              textLocalize('recall_agent_panel_hint'),
              style: TextStyle(
                color: widget.isDark
                    ? Colors.white.withValues(alpha: 0.62)
                    : BDDesign.colorMutedBlue.withValues(alpha: 0.88),
                fontSize: 13,
              ),
            ),
          ),
        ],
      ],
    );
  }
}

class RecallSearchModeButton extends StatelessWidget {
  final bool isDark;
  final IconData icon;
  final VoidCallback onTap;

  const RecallSearchModeButton({
    super.key,
    required this.isDark,
    required this.icon,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    const buttonRadius = BorderRadius.all(Radius.circular(10));
    final foreground = isDark
        ? Colors.white.withValues(alpha: 0.82)
        : BDDesign.colorInkBlack;

    return Tooltip(
      message: textLocalize('recall_search_mode'),
      child: Material(
        color: Colors.transparent,
        shape: const RoundedRectangleBorder(borderRadius: buttonRadius),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          customBorder: const RoundedRectangleBorder(
            borderRadius: buttonRadius,
          ),
          splashFactory: NoSplash.splashFactory,
          overlayColor: const WidgetStatePropertyAll<Color>(Colors.transparent),
          child: Container(
            height: 32,
            width: 32,
            decoration: BoxDecoration(
              color: isDark
                  ? Colors.white.withValues(alpha: 0.06)
                  : BDDesign.colorMutedBlue.withValues(alpha: 0.08),
              borderRadius: buttonRadius,
              border: Border.all(
                color: isDark
                    ? Colors.white.withValues(alpha: 0.08)
                    : BDDesign.colorMutedBlue.withValues(alpha: 0.18),
              ),
            ),
            child: Center(child: Icon(icon, size: 16, color: foreground)),
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
          child: SingleChildScrollView(
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
                  textLocalize('recall_search_mode_desc'),
                  style: TextStyle(
                    color: hintColor,
                    fontSize: 12.5,
                    height: 1.35,
                  ),
                ),
                const SizedBox(height: 16),
                modeTile(
                  mode: RecallSearchMode.cloud,
                  icon: Icons.cloud_rounded,
                ),
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
      ),
    );
  }
}
