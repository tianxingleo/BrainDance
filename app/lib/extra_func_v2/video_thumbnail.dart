import 'package:video_thumbnail_pro/video_thumbnail_pro.dart';
import 'package:video_thumbnail_pro/index.dart';
import 'package:braindance/extra_func/dir_and_file.dart';
import 'package:path/path.dart' as path;

class VThumb {
  static Future<String> ensureThumb(String videoPath) async {
    final fname = "${path.basenameWithoutExtension(videoPath)}.jpg";
    final pathThumb = path.join(await DirFinder.cacheDir(), "thumbNails");
    final pathThumbFull = path.join(pathThumb, fname);
    if (await FileSystem.checkFileExists(pathThumbFull)) {
      return pathThumbFull;
    }

    final thumb = "assets/sprites/video-camera.png";
    if (pathThumb.isEmpty) {
      return thumb;
    }
    await DirSystem.ensureDir(pathThumb);
    try {
      await VideoThumbnailPro.thumbnailFile(
        video: videoPath,
        imageFormat: ImageFormat.JPEG, // 可以选择PNG或JPEG
        quality: 50, // 质量，范围0-100
        timeMs: 0, // 生成缩略图的时间点，单位毫秒（1秒）
        thumbnailPath: pathThumb,
      );
    } catch (e) {
      return thumb;
    }
    if (await FileSystem.checkFileExists(pathThumbFull)) {
      return pathThumbFull;
    }
    return thumb;
  }
}
