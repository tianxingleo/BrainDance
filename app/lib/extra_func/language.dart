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
      "recall": "过往回忆",
      "record": "相机记录",
      "generate": "图文生成",
      "settings": "设置",
      "tip_unava": "该功能尚未开放，敬请期待！",
      "tip_oversize": "上传的文件太大了！",
      "tip_overquan": "你上传的文件太多了！",
      "tip_cache": "已成功清除缓存",
      "tip_fail" : "文件上传失败！",
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
      "recall": "Recall",
      "record": "Record",
      "generate": "Generate",
      "settings": "Settings",
      "tip_unava": "This feature is not yet available, stay tuned!",
      "tip_oversize": "Uploaded file(s) are too large!",
      "tip_overquan": "You have uploaded too many files!",
      "tip_cache": "The cache is successfully cleared",
      "tip_fail" : "Failed to upload file(s)!",
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
