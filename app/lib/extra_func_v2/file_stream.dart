import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:path/path.dart' as path_joiner;
class FileStream {
  static Future<List<String>> appLoad(Future<String> futDir, String fname) async {//返回内容
    final String dir = await futDir;
    final String path = path_joiner.join(dir, fname);
    if (await FileSystem.checkFileExists(path)) {//检测文件是否存在
      return await FileSystem.readFile(path);
    }
    return List.empty();
  }
  static Future<void> appSave(Future<String> futDir, String fname, List<String> lines) async {
    final String dir = await futDir;
    final String path = path_joiner.join(dir, fname);
    await DirSystem.ensureDir(dir);
    await FileSystem.writeFile(path, lines);
  }
  static Future<void> appDel(Future<String> futDir, String fname) async {
    final String dir = await futDir;
    final String path = path_joiner.join(dir, fname);
    await FileSystem.deleteFile(path);
  }
  static Future<String> appGetPath(Future<String> futDir, String fname) async {
    final String dir = await futDir;
    final String path = path_joiner.join(dir, fname);
    if (await DirSystem.ensureDir(dir)) return path;
    return '';
  }
}