import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:path/path.dart' as path_joiner;

//今后需要添加AppDir，第1/2步：从enum添加
enum AppDir { support, cache }

//代码部分
class FileStream {
  //自动根据枚举长度生成列表，进行缓存
  static final List<String> dirs = List.filled(AppDir.values.length, '');
  static Future<String> getDir(AppDir name) async {
    //根据名称缓存列表
    //第2/2步：修改switch代码, 3个相同数字/1个路径
    switch (name) {
      case AppDir.support:
        if (dirs[0].isEmpty) {
          dirs[0] = await DirFinder.supportDir();
        }
        return dirs[0];
      case AppDir.cache:
        if (dirs[1].isEmpty) {
          dirs[1] = await DirFinder.cacheDir();
        }
        return dirs[1];
    }
  }

  static Future<List<String>> appLoad(AppDir dirName, String fname) async {
    //返回内容
    final String dir = await getDir(dirName);
    final String path = path_joiner.join(dir, fname);
    if (await FileSystem.checkFileExists(path)) {
      //检测文件是否存在
      return await FileSystem.readFile(path);
    }
    return List.empty();
  }

  static Future<void> appSave(
    AppDir dirName,
    String fname,
    List<String> lines,
  ) async {
    final String dir = await getDir(dirName);
    final String path = path_joiner.join(dir, fname);
    await DirSystem.ensureDir(dir);
    await FileSystem.writeFile(path, lines);
  }

  static Future<void> appDel(AppDir dirName, String fname) async {
    final String dir = await getDir(dirName);
    final String path = path_joiner.join(dir, fname);
    await FileSystem.deleteFile(path);
  }
}
