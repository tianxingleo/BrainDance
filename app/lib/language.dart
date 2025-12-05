class Localize {
  static const Map<String, String> mapZh = {
    "lang" : "zh_cn",
    "lang_name" : "简体中文",
    "title" : "Brain Dance",
    "home_page" : "Brain Dance主界面",
    "main" : "你已推动按钮这么多次：",
  };
  static const Map<String, String> mapEn = {
    "lang" : "en_us",
    "lang_name" : "English",
    "title" : "Brain Dance",
    "home_page" : "Brain Dance Home Page",
    "main" : "You have pushed the button this many times:",
  };
  static const List<Map<String, String>> languageMaps = [mapZh, mapEn];

  static String t({int langCode = 0, String text = ""}) {
    return languageMaps[langCode][text] ?? text;
  }
}