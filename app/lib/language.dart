enum Language {
  zh (
    name : "简体中文",
    map : {
      "locale" : "zh_CN",
      "title" : "流光 · 记",
      "home_page" : "流光 · 记主界面",
      "main" : "你已推动按钮这么多次：",
      "main_2" : "切换语言并保存",
    },
  ),
  en (
    name : "English",
    map : {
      "locale" : "en_US",
      "title" : "Brain Dance",
      "home_page" : "Brain Dance Home Page",
      "main" : "You have pushed the button this many times:",
      "main_2" : "Change Language and Save",
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