import 'dart:io';

void main() {
  var file = File('c:/Projects/VibeBrainDance/BrainDance/app/lib/pages/recall.dart');
  var content = file.readAsStringSync();

  content = content.replaceAll(
    '''  // 更黑的夜间色值
  final darkBg = const Color(0xFF101014);
  final darkCard = const Color(0xFF18181C);
  final darkInput = const Color(0xFF23232A);
  final darkBorder = const Color(0xFF23232A);''',
    ''''''
  );

  content = content.replaceAll('const Color(0xFF101014)', 'theme.grayColor14');
  content = content.replaceAll('const Color(0xFF18181C)', 'theme.grayColor13');
  content = content.replaceAll('const Color(0xFF23232A)', 'theme.grayColor12');
  
  content = content.replaceAll('isDark ? const Color(0xFFFFFFFF) : const Color(0xFF333333)', 'isDark ? theme.whiteColor1 : theme.fontGyColor1');
  content = content.replaceAll('isDark ? const Color(0xFFEEEEEE) : const Color(0xFF333333)', 'isDark ? theme.grayColor3 : theme.fontGyColor1');
  content = content.replaceAll('isDark ? const Color(0xFFCCCCCC) : theme.fontGyColor3', 'isDark ? theme.fontGyColor4 : theme.fontGyColor3');

  content = content.replaceAll('color: isDark ? const Color(0xFF888888) : theme.fontGyColor3,', 'color: isDark ? theme.fontGyColor4 : theme.fontGyColor3,');
  content = content.replaceAll('color: isDark ? const Color(0xFFFFFFFF) : const Color(0xFF333333),', 'color: isDark ? theme.whiteColor1 : theme.fontGyColor1,');
  content = content.replaceAll('color: isDark ? const BorderSide(color: Color(0xFF4582FF), width: 1.5) : BorderSide(color: theme.brandColor7, width: 1.5)', 'borderSide: isDark ? BorderSide(color: theme.brandColor8, width: 1.5) : BorderSide(color: theme.brandColor7, width: 1.5)');

  content = content.replaceAll('const Color(0xFFFFFFFF)', 'theme.whiteColor1');
  content = content.replaceAll('Colors.white', 'theme.whiteColor1');
  
  // replace the variable usages
  content = content.replaceAll('darkBg', 'theme.grayColor14');
  content = content.replaceAll('darkCard', 'theme.grayColor13');
  content = content.replaceAll('darkInput', 'theme.grayColor12');
  content = content.replaceAll('darkBorder', 'theme.grayColor11');

  file.writeAsStringSync(content);
}

