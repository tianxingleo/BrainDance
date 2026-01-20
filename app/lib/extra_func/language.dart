enum Language {
  zh (
    name : "简体中文",
    map : {
      "lang" : "简体中文",
      "locale" : "zh_CN",
      "title" : "流光 · 记",
      "home_page" : "流光 · 记主界面",
      "pick_null" : "请选择",
      "set_lang" : "当前语言（点击切换）",
      "set_night" : "夜间模式",
      "set_ver" : "版本号",
      "set_pub" : "发布日期",
      "set_tab1" : "显示",
      "set_tab2" : "视频",
      "set_tab3" : "关于",
      "set_tab4" : "帮助",
      "gen_top" : "生成自",
      "gen_pic" : "图片",
      "gen_text" : "文本",
      "gen_label_pic" : "使用图片：",
      "gen_label_text" : "使用文本：",
      "gen_tip_pic" : "请选择图片文件 (小于4MB)",
      "gen_tip_text" : "请输入描述文本",
      "gen_button" : "生成记忆",
      "recall" : "过往回忆",
      "record" : "相机记录",
      "generate" : "图文生成",
      "settings" : "设置",
      "tip_unava" : "该功能尚未开放，敬请期待！",
      "tip_oversize" : "上传的图片太大了！",
      "tip_overquan" : "你上传的图片太多了！",
    },
  ),
  en (
    name : "English",
    map : {
      "lang" : "English",
      "locale" : "en_US",
      "title" : "Brain Dance",
      "home_page" : "Brain Dance Home Page",
      "pick_null" : "Please select",
      "set_lang" : "Language (Tap to switch)",
      "set_night" : "Night Mode",
      "set_ver" : "Version",
      "set_pub" : "Publish Date",
      "set_tab1" : "Display",
      "set_tab2" : "Video",
      "set_tab3" : "About",
      "set_tab4" : "Help",
      "gen_top" : "Generate from",
      "gen_pic" : "Image",
      "gen_text" : "Text",
      "gen_label_pic" : "Use Image:",
      "gen_label_text" : "Use Text:",
      "gen_tip_pic" : "Please select an image file (less than 4MB)",
      "gen_tip_text" : "Please enter a description text",
      "gen_button" : "Generate Memory",
      "recall" : "Recall",
      "record" : "Record",
      "generate" : "Generate",
      "settings" : "Settings",
      "tip_unava" : "This feature is not yet available, stay tuned!",
      "tip_oversize" : "Uploaded image(s) are too large!",
      "tip_overquan" : "You have uploaded too many images!",
    },
  );
  const Language({
    required this.name,
    required this.map,
  });
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