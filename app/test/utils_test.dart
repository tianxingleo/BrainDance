/// Smoke tests: 确认 dir_and_file.dart 的纯 I/O 工具函数能正常工作。
///
/// 这些测试使用系统临时目录，不依赖网络或硬件。
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:braindance/extra_func/dir_and_file.dart';

void main() {
  late String tempRoot;

  setUp(() async {
    // 每个测试使用独立的临时目录，避免互相干扰
    tempRoot =
        '${Directory.systemTemp.path}/braindance_smoke_${DateTime.now().microsecondsSinceEpoch}';
  });

  tearDown(() async {
    try {
      final dir = Directory(tempRoot);
      if (await dir.exists()) {
        await dir.delete(recursive: true);
      }
    } catch (_) {
      // 忽略清理错误
    }
  });

  // ─── DirSystem ─────────────────────────────────────────────────────
  group('DirSystem 目录操作', () {
    test('checkDirExists - 不存在的目录返回 false', () async {
      expect(await DirSystem.checkDirExists(tempRoot), isFalse);
    });

    test('createDir - 成功创建目录', () async {
      final result = await DirSystem.createDir(tempRoot);
      expect(result, isTrue);
      expect(await Directory(tempRoot).exists(), isTrue);
    });

    test('ensureDir - 首次调用创建目录', () async {
      final result = await DirSystem.ensureDir(tempRoot);
      expect(result, isTrue);
      expect(await Directory(tempRoot).exists(), isTrue);
    });

    test('ensureDir - 第二次调用幂等', () async {
      await DirSystem.createDir(tempRoot);
      final result = await DirSystem.ensureDir(tempRoot);
      expect(result, isTrue);
    });

    test('deleteDir - 删除已创建的目录', () async {
      await DirSystem.createDir(tempRoot);
      final result = await DirSystem.deleteDir(tempRoot);
      expect(result, isTrue);
      expect(await Directory(tempRoot).exists(), isFalse);
    });

    test('createDir - 支持嵌套路径', () async {
      final nested = '$tempRoot/a/b/c';
      final result = await DirSystem.createDir(nested);
      expect(result, isTrue);
      expect(await Directory(nested).exists(), isTrue);
    });
  });

  // ─── FileSystem ────────────────────────────────────────────────────
  group('FileSystem 文件操作', () {
    late String filePath;

    setUp(() async {
      await DirSystem.createDir(tempRoot);
      filePath = '$tempRoot/smoke_test.txt';
    });

    test('checkFileExists - 不存在的文件返回 false', () async {
      expect(await FileSystem.checkFileExists(filePath), isFalse);
    });

    test('writeFile + readFile - 写入后读取内容一致', () async {
      final lines = ['hello', 'world', 'braindance'];
      final writeOk = await FileSystem.writeFile(filePath, lines);
      expect(writeOk, isTrue);

      final readLines = await FileSystem.readFile(filePath);
      expect(readLines.length, 3);
      expect(readLines[0], 'hello');
      expect(readLines[1], 'world');
      expect(readLines[2], 'braindance');
    });

    test('checkFileExists - 写入后返回 true', () async {
      await FileSystem.writeFile(filePath, ['content']);
      expect(await FileSystem.checkFileExists(filePath), isTrue);
    });

    test('deleteFile - 删除后文件不存在', () async {
      await FileSystem.writeFile(filePath, ['data']);
      final delOk = await FileSystem.deleteFile(filePath);
      expect(delOk, isTrue);
      expect(await FileSystem.checkFileExists(filePath), isFalse);
    });

    test('readFile - 空文件读取', () async {
      // 写空内容
      await FileSystem.writeFile(filePath, []);
      final lines = await FileSystem.readFile(filePath);
      // "".split("\n") => [""]
      expect(lines, isNotEmpty);
    });

    test('readFile - 不存在的文件返回 fallback', () async {
      final lines = await FileSystem.readFile('/nonexistent/path/file.txt');
      expect(lines, isNotEmpty);
      expect(lines.first, isEmpty);
    });
  });
}
