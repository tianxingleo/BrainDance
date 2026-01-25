enum Language {
  zh(
    name: "简体中文",
    map: {
      "lang": "简体中文",
      "locale": "zh_CN",
      "title": "流光 · 记",
      "home_page": "流光 · 记主界面",
      "pick_null": "请选择",
      "set_lang": "当前语言（点击切换）",
      "set_night": "夜间模式",
      "set_ver": "版本号",
      "set_pub": "发布日期",
      "set_cache": "清除缓存",
      "set_tab1": "显示",
      "set_tab2": "视频",
      "set_tab3": "应用",
      "set_tab4": "帮助",
      "gen_top": "生成自",
      "gen_pic": "图片",
      "gen_text": "文本",
      "gen_video": "视频",
      "gen_tip_pic": "请选择图片文件 (小于4MB)",
      "gen_tip_text": "请输入描述文本",
      "gen_tip_video": "请选择视频文件\n(清晰度小于等于 1080p 30fps，\n拍摄时长 3min 以内)",
      "gen_tip_textbox": "在此处键入文本",
      "gen_button": "生成记忆",
      "gen_shot": "拍照",
      "gen_gallery": "从相册中选择",
      "gen_cancel": "取消",
      "reco_tip_title1" : "一、设备与设置",
      "reco_tip_title2" : "二、拍摄环境与操作",
      "reco_tip_title3" : "三、后期检查",
      "reco_tip1" : """设备：使用支持1080p/30fps录像的智能手机或相机。
稳定：建议使用三脚架，避免画面抖动。
分辨率与帧率：在设置中手动选择 1920×1080（1080p） 和 30fps。
曝光：使用自动曝光模式。如光线明暗变化大，可手动锁定曝光。
对焦：使用自动连续对焦模式。""",
      "reco_tip2" : """光线：在光线均匀、充足的条件下拍摄。避免逆光（如背景是窗户）。
动作：拍摄时，尽量避免快速移动或晃动相机。
内容：规划不同的拍摄内容和角度，避免画面重复。""",
      "reco_tip3" : """如视频使用手机自带相机软件拍摄，请用剪辑软件（如剪映）检查视频，剪掉抖动、模糊或过曝的片段。
最终导出时，确认参数为 1080p 和 30fps。""",
      "reco_camun": "相机访问被拒绝。\n请在系统设置中启用相机权限。",
      "reco_wait": "相机初始化中...",
      "recall": "过往回忆",
      "record": "相机记录",
      "generate": "图文生成",
      "settings": "设置",
      "tip_unava": "该功能尚未开放，敬请期待！",
      "tip_oversize": "上传的文件太大了！",
      "tip_overquan": "你上传的文件太多了！",
      "tip_cache": "已成功清除缓存",
      "tip_fail": "文件上传失败！",
      "tip_no_permission" : "应用没有权限，无法保存到图库。",
    },
  ),
  en(
    name: "English",
    map: {
      "lang": "English",
      "locale": "en_US",
      "title": "Brain Dance",
      "home_page": "Brain Dance Home Page",
      "pick_null": "Please select",
      "set_lang": "Language (Tap to switch)",
      "set_night": "Night Mode",
      "set_ver": "Version",
      "set_pub": "Publish Date",
      "set_cache": "Clear Cache",
      "set_tab1": "Display",
      "set_tab2": "Video",
      "set_tab3": "App",
      "set_tab4": "Help",
      "gen_top": "Generate from",
      "gen_pic": "Image",
      "gen_text": "Text",
      "gen_video": "Video",
      "gen_tip_pic": "Please select an image file (less than 4MB)",
      "gen_tip_text": "Please enter a description text",
      "gen_tip_video":
          "Please select a video file\n(Resolution ≤ 1080p 30fps,\nDuration within 3 minutes)",
      "gen_tip_textbox": "Type text here",
      "gen_button": "Generate Memory",
      "gen_shot": "Use Camera",
      "gen_gallery": "Choose from Gallery",
      "gen_cancel": "Cancel",
      "reco_tip_title1" : "1. Equipment & Settings",
      "reco_tip_title2" : "2. Shooting Environment & Operation",
      "reco_tip_title3" : "3. Post-Shooting Check",
      "reco_tip1" : """Device: Use a smartphone or camera that supports 1080p/30fps video recording.
Stability: It is recommended to use a tripod to avoid shaky footage.
Resolution & Frame Rate: Manually select 1920x1080 (1080p) and 30fps in the settings.
Exposure: Use auto-exposure mode. If there are significant changes in lighting, manually lock the exposure.
Focus: Use continuous auto-focus mode.""",
      "reco_tip2" : """Lighting: Shoot under even and sufficient lighting. Avoid backlighting (e.g., having a window as the background).
Movement: During shooting, try to avoid rapid movements or shaking of the camera.
Content: Plan diverse shooting content and angles to avoid repetitive footage.""",
      "reco_tip3" : """If the video was shot using a phone's built-in camera app, use editing software (e.g., CapCut) to review the video and trim any shaky, blurry, or overexposed segments.
When finally exporting, confirm the settings are 1080p and 30fps.""",
      "reco_camun":
          "Cannot access the camera.\nPlease check and grant the camera permission in your device settings to continue.",
      "reco_wait": "Initializing the camera...",
      "recall": "Recall",
      "record": "Record",
      "generate": "Generate",
      "settings": "Settings",
      "tip_unava": "This feature is not yet available, stay tuned!",
      "tip_oversize": "Uploaded file(s) are too large!",
      "tip_overquan": "You have uploaded too many files!",
      "tip_cache": "The cache is successfully cleared",
      "tip_fail": "Failed to upload file(s)!",
      "tip_no_permission" : "Does not have permission to save to the gallery.",
    },
  );

  const Language({required this.name, required this.map});
  final String name;
  final Map<String, String> map;
}

class Localize {
  static Map<String, String> getLangMap(String localeCode) {
    for (var lang in Language.values) {
      if (lang.map['locale'] == localeCode) {
        return lang.map;
      }
    }
    return Language.en.map; // Default to English if not found
  }
}
