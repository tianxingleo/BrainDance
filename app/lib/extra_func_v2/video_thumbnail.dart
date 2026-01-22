import 'package:video_thumbnail_pro/video_thumbnail_pro.dart';
import 'package:video_thumbnail_pro/index.dart';
import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:path/path.dart' as path;

class VThumb {
  static Future<String> ensureThumb(String videoPath) async {
    var thumb = "assets/sprites/video-camera.png";
    if (!await VThumb.hasThumb(videoPath)) {
      var temp = await VThumb.generate(videoPath);
      if (temp.isNotEmpty) {
        thumb = temp;
      }
    } else {
      thumb = await VThumb.getPath(videoPath);
    }
    return thumb;
  }

  static Future<bool> hasThumb(String videoPath) async {
    final pathThumb = await getPath(videoPath);
    return await FileSystem.checkFileExists(pathThumb);
  }

  static Future<String> getPath(String videoPath) async {
    final fname = "${path.basenameWithoutExtension(videoPath)}.jpg";
    return path.join(await DirFinder.cacheDir(), fname);
  }

  static Future<String> generate(String videoPath) async {
    final pathThumb = await DirFinder.cacheDir();
    await DirSystem.ensureDir(pathThumb);
    final outputFPath = path.join(pathThumb, "${path.basenameWithoutExtension(videoPath)}.jpg");
    if (pathThumb.isEmpty) {
      return '';
    }
    try {
      await VideoThumbnailPro.thumbnailFile(
        video: videoPath,
        imageFormat: ImageFormat.JPEG, // 可以选择PNG或JPEG
        quality: 50, // 质量，范围0-100
        timeMs: 0, // 生成缩略图的时间点，单位毫秒（1秒）
        thumbnailPath: pathThumb,
      );
    } catch (e) {
      return '';
    }
    if (await FileSystem.checkFileExists(outputFPath)) {
      return outputFPath;
    }
    return '';
  }
}
