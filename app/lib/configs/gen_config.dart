import 'package:tdesign_flutter/tdesign_flutter.dart';
import '../extra_func_v2/file_stream.dart';

class GenConfig {
  static const imagePathsFileName = "genImagePaths.txt";
  static const videoPathsFileName = "genVideoPaths.txt";
  static const textFileName = "genText.txt";
  static List<TDUploadFile> uploadedImages = [];
  static List<TDUploadFile> uploadedVideos = [];
  static String uploadedText = "";

  static void saveUploadedAssets() {
    //Image
    List<String> imagePaths = [];
    for (TDUploadFile file in uploadedImages) {
      imagePaths.add(file.assetPath.toString());
    }
    if (imagePaths.isNotEmpty) {
      saveImagePathsFile(imagePaths);
    } else {
      deleteImagePathsFile();
    }
    //Text
    if (uploadedText.isNotEmpty) {
      saveTextFile(uploadedText);
    } else {
      deleteTextFile();
    }
    //Video
    List<String> videoPaths = [];
    for (TDUploadFile file in uploadedVideos) {
      videoPaths.add(file.assetPath.toString());
    }
    if (videoPaths.isNotEmpty) {
      saveVideoPathsFile(videoPaths);
    } else {
      deleteVideoPathsFile();
    }
  }

  static Future<List<String>> loadImagePathsFile() async {
    return await FileStream.appLoad(AppDir.cache, imagePathsFileName);
  }

  static Future<String> loadTextFile() async {
    final List<String> result = await FileStream.appLoad(
      AppDir.cache,
      textFileName,
    );
    return result.join();
  }

  static Future<List<String>> loadVideoPathsFile() async {
    return await FileStream.appLoad(AppDir.cache, videoPathsFileName);
  }

  static Future<void> saveImagePathsFile(List<String> paths) async {
    await FileStream.appSave(AppDir.cache, imagePathsFileName, paths);
  }

  static Future<void> saveTextFile(String text) async {
    await FileStream.appSave(AppDir.cache, textFileName, [text]);
  }

  static Future<void> saveVideoPathsFile(List<String> paths) async {
    await FileStream.appSave(AppDir.cache, videoPathsFileName, paths);
  }

  static Future<void> deleteImagePathsFile() async {
    await FileStream.appDel(AppDir.cache, imagePathsFileName);
  }

  static Future<void> deleteTextFile() async {
    await FileStream.appDel(AppDir.cache, textFileName);
  }

  static Future<void> deleteVideoPathsFile() async {
    await FileStream.appDel(AppDir.cache, videoPathsFileName);
  }
}
