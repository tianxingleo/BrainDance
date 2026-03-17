  @override
  Widget build(BuildContext context) {
    final theme = TDTheme.of(context);
    final isDark = AppConfig.isNightMode;
    final textColor = isDark ? const Color(0xFFFFFFFF) : BDDesign.colorInkBlack;
    final iconColor = isDark ? const Color(0xFFEEEEEE) : BDDesign.colorMutedBlue;
    final bgCardColor = isDark ? const Color(0xFF1C1C1E) : BDDesign.colorPaperWhite;

    final List<Widget> tabContents = [
      Container(
        color: Colors.transparent,
        child: TDTabBar(
          tabs: [
            TDTab(text: textLocalize('gen_pic')),
            TDTab(text: textLocalize('gen_text')),
            TDTab(text: textLocalize('gen_video')),
          ],
          controller: _tabController,
          showIndicator: true,
          indicatorPadding: const EdgeInsets.all(4.0),
          indicatorWidth: 24,
          indicatorHeight: 3,
          indicatorColor: BDDesign.colorMutedBlue,
          onTap: (index) {
            setState(() {});
          },
          labelStyle: tabTextStyle.copyWith(
            fontWeight: FontWeight.w600,
            color: BDDesign.colorMutedBlue,
          ),
          unselectedLabelStyle: tabTextStyle.copyWith(
            fontWeight: FontWeight.w400,
            color: isDark ? const Color(0xFF888888) : theme.fontGyColor3,
          ),
        ),
      ),
    ];
    Widget? currentTabContent;
    switch (_tabController.index) {
      case 0:
        final TDUpload myTDUpload = TDUpload(
          key: _uploadKey,
          files: GenConfig.uploadedImages,
          multiple: true,
          max: maxImageCount,
          onUploadTap: () {
            _showActionSheet(context, true);
          },
          onChange: (files, type) {
            switch (type) {
              case TDUploadType.add:
                GenConfig.uploadedImages = [
                  ...GenConfig.uploadedImages,
                  ...files,
                ];
                break;
              case TDUploadType.remove:
                for (var f in files) {
                  GenConfig.uploadedImages.remove(f);
                }
                break;
              case TDUploadType.replace:
                break;
            }
            setState(() {
              _uploadKey = UniqueKey();
            });
          },
          mediaType: [TDUploadMediaType.image],
          width: 150,
          height: 150,
        );
        currentTabContent = Scrollbar(
          key: const ValueKey<int>(0),
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  textLocalize('gen_tip_pic'),
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: textColor,
                  ),
                ),
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: bgCardColor,
                    borderRadius: BDDesign.radiusNormal,
                    boxShadow: [
                      if (!isDark) BDDesign.shadowLight,
                    ],
                  ),
                  child: myTDUpload,
                ),
                const SizedBox(height: 100),
              ],
            ),
          ),
        );
        break;
      case 1:
        _textEditingController.text = GenConfig.uploadedText;
        currentTabContent = Scrollbar(
          key: const ValueKey<int>(1),
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  textLocalize('gen_tip_text'),
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: textColor,
                  ),
                ),
                const SizedBox(height: 24),
                Container(
                  decoration: BoxDecoration(
                    color: bgCardColor,
                    borderRadius: BDDesign.radiusNormal,
                    boxShadow: [
                      if (!isDark) BDDesign.shadowLight,
                    ],
                  ),
                  child: TDTextarea(
                    controller: _textEditingController,
                    hintText: textLocalize('gen_tip_textbox'),
                    minLines: 8,
                    maxLines: 20,
                    onChanged: (value) {
                      GenConfig.uploadedText = value;
                    },
                    decoration: BoxDecoration(
                      color: Colors.transparent,
                      borderRadius: BDDesign.radiusNormal,
                      border: Border.all(color: Colors.transparent),
                    ),
                    textStyle: TextStyle(
                      color: textColor,
                      fontSize: 16,
                    ),
                  ),
                ),
                const SizedBox(height: 100),
              ],
            ),
          ),
        );
        break;
      case 2:
        final TDUpload myTDUpload = TDUpload(
          type: TDUploadBoxType.circle,
          key: _uploadKey2,
          files: GenConfig.uploadedVideos,
          multiple: false,
          onUploadTap: () {
            _showActionSheet(context, false);
          },
          onChange: (files, type) {
            switch (type) {
              case TDUploadType.add:
                GenConfig.uploadedVideos = [
                  ...GenConfig.uploadedVideos,
                  ...files,
                ];
                break;
              case TDUploadType.remove:
                for (var f in files) {
                  GenConfig.uploadedVideos.remove(f);
                }
                break;
              case TDUploadType.replace:
                break;
            }
            setState(() {
              _uploadKey2 = UniqueKey();
            });
          },
          mediaType: [TDUploadMediaType.video],
          width: 150,
          height: 150,
        );
        currentTabContent = Scrollbar(
          key: const ValueKey<int>(2),
          controller: _scrollController,
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  textLocalize('gen_tip_video'),
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: textColor,
                  ),
                ),
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: bgCardColor,
                    borderRadius: BDDesign.radiusNormal,
                    boxShadow: [
                      if (!isDark) BDDesign.shadowLight,
                    ],
                  ),
                  child: myTDUpload,
                ),
                const SizedBox(height: 100),
              ],
            ),
          ),
        );
        break;
    }

    if (currentTabContent != null) {
      tabContents.add(
        Expanded(
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            switchInCurve: Curves.easeOutCubic,
            switchOutCurve: Curves.easeInCubic,
            transitionBuilder: (Widget child, Animation<double> animation) {
              return FadeTransition(
                opacity: animation,
                child: SlideTransition(
                  position: Tween<Offset>(
                    begin: const Offset(0.05, 0.0),
                    end: Offset.zero,
                  ).animate(animation),
                  child: child,
                ),
              );
            },
            child: currentTabContent,
          ),
        ),
      );
    }
    
    return Scaffold(
      backgroundColor: isDark ? const Color(0xFF101014) : BDDesign.colorAshGray,
      body: SafeArea(
        child: Stack(
          children: [
            Column(
              children: [
                // 顶部标题对齐 Recall 风格
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        textLocalize('gen_top'),
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.w600,
                          color: textColor,
                        ),
                      ),
                    ],
                  ),
                ),
                Expanded(child: Column(children: tabContents)),
              ],
            ),
            Align(
              alignment: const Alignment(0, 0.9),
              child: Container(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: TDButton(
                  onTap: (_isUploading || _isGenerating) ? () {} : _submit,
                  style: TDButtonStyle(
                    backgroundColor: isDark ? const Color(0xFF2A2A2E) : BDDesign.colorMutedBlue,
                    textColor: Colors.white,
                    radius: BDDesign.radiusNormal,
                  ),
                  type: TDButtonType.fill,
                  shape: TDButtonShape.rectangle,
                  theme: TDButtonTheme.primary,
                  size: TDButtonSize.large,
                  width: MediaQuery.of(context).size.width * 0.85,
                  text: (_isUploading || _isGenerating) 
                     ? (_isGenerating ? textLocalize('gen_text_generating') : '正在上传... $((_uploadProgress * 100).toStringAsFixed(1))%') 
                     : textLocalize('gen_button'),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
