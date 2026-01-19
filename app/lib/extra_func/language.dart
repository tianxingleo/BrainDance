enum Language {
  zh (
    name : "简体中文",
    map : {
      "lang" : "简体中文",
      "locale" : "zh_CN",
      "title" : "流光 · 记",
      "home_page" : "流光 · 记主界面",
      "set_lang" : "当前语言（点击切换）",
      "set_night" : "夜间模式",
      "set_tab1" : "显示",
      "set_tab2" : "视频",
      "set_tab3" : "关于",
      "set_tab4" : "帮助",
      "recall" : "过往回忆",
      "record" : "相机记录",
      "generate" : "图文生成",
      "settings" : "设置",
    },
  ),
  en (
    name : "English",
    map : {
      "lang" : "English",
      "locale" : "en_US",
      "title" : "Brain Dance",
      "home_page" : "Brain Dance Home Page",
      "set_lang" : "Language (Tap to switch)",
      "set_night" : "Night Mode",
      "set_tab1" : "Display",
      "set_tab2" : "Video",
      "set_tab3" : "About",
      "set_tab4" : "Help",
      "recall" : "Recall",
      "record" : "Record",
      "generate" : "Generate",
      "settings" : "Settings",
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