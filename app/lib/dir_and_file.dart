import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path_joiner;
import 'dart:io';
//All functions here is safe to use.
class DirFinder {
  //Part 1: Find Directories
  //Notice: DownloadsDir can be nothing!
  static Future<String> documentsDir() async {
    try
    {
      return await getApplicationDocumentsDirectory().then((value) {
        if (Platform.isWindows) {
          return path_joiner.join(value.path,"BrainDance");//If projectName changes, it should be changed.
        }
        return value.path;
      });
    }
    catch (e)
    {
      return "";
    }
  }
  static Future<String> cacheDir() async {
    try
    {
      return await getApplicationCacheDirectory().then((value) => value.path);
    }
    catch (e)
    {
      return "";
    }
  }
  static Future<String> supportDir() async {
    try
    {
      return await getApplicationSupportDirectory().then((value) => value.path);
    }
    catch (e)
    {
      return "";
    }
  }
  static Future<String> downloadsDir() async {
    try
    {
      return await getDownloadsDirectory().then((value) {
        return (value == null) ? "" : value.path;
      });
    }
    catch (e)
    {
      try
      {
        return await getTemporaryDirectory().then((value) {
          return value.path;
        });
      }
      catch (e)
      {
        return "";
      }
    }
  }
}
class DirSystem {
  //Part 2: Check & Do Directories
  static Future<bool> checkDirExists(String path) async {
    final dir = Directory(path);
    try
    {
      return await dir.exists();
    }
    catch (e)
    {
      return false;
    }
  }
  static Future<bool> createDir(String path) async {
    final dir = Directory(path);
    try
    {
      await dir.create(recursive: true);
      return true;
    }
    catch (e)
    {
      return false;
    }
  }
  static Future<bool> ensureDir(String path) async {
    bool exists = await checkDirExists(path);
    if (!exists)
    {
      return await createDir(path);
    }
    return true;
  }
  static Future<bool> deleteDir(String path) async {
    final dir = Directory(path);
    try
    {
      await dir.delete(recursive: true);
      return true;
    }
    catch (e)
    {
      return false;
    }
  }
}
class FileSystem {
  //Part 3: File I/O
  static Future<bool> checkFileExists(String path) async {
    final file = File(path);
    try
    {
      return await file.exists();
    }
    catch (e)
    {
      return false;
    }
  }
  static Future<List<String>> readFile(String path) async {
      final file = File(path);
      String content;
      try
      {
        content = await file.readAsString();
      }
      catch (e)
      {
        content = "";
      }
      return content.replaceAll('\r','').split("\n");
  }
  static Future<bool> writeFile(String path, List<String> lines) async {
    final file = File(path);
    String content = lines.join("\n");
    try
    {
      await file.writeAsString(content);
      return true;
    }
    catch (e)
    {
      return false;
    }
  }
  static Future<bool> deleteFile(String path) async {
    final file = File(path);
    try
    {
      await file.delete();
      return true;
    }
    catch (e)
    {
      return false;
    }
  }
}