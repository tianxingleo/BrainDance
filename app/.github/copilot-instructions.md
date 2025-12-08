## 快速目标

- 让 AI 编码代理快速上手并安全修改此 Flutter 应用（多平台）。

## 项目概览（大图）

- 这是一个标准的 Flutter 多平台应用，根目录包含 `android/`, `ios/`, `windows/`, `linux/`, `macos/` 与 `web/`：构建/运行请使用 `flutter` 工具链。
- 逻辑主要位于 `lib/`：入口 `lib/main.dart`，简单的本地化封装在 `lib/language.dart`。
- 配置常量集中在 `AppConfig`（见 `lib/main.dart`），主题与国际化通过静态方法/常量共享。

## 关键文件与示例

- `lib/main.dart`: 应用入口。注意 `AppConfig` 用于共享颜色/版本/语言选择；文本取用 `textLocalize("id")`，该函数内部调用 `Localize.t(...)`。
  例如：`textLocalize("title")` → 由 `lib/language.dart` 的 `Localize.languageMaps` 提供字符串。
- `lib/language.dart`: 简单的静态本地化实现（Map 列表 + `t({int langCode, String text})`）。不要假设使用 intl 包；直接字符串键/值映射。
- `pubspec.yaml`: 当前依赖非常精简，仅含 `cupertino_icons`。修改依赖后请运行 `flutter pub get`。

## 项目约定与风格（可自动化/要遵循的规则）

- 本项目倾向于使用静态工具类（例如 `AppConfig`, `Localize`）用于全局配置/字符串。尽量遵循这一模式而不是引入全局单例状态。
- 本地化以字符串 id 为中心（非 arb/intl）。修改/新增文本时，更新 `lib/language.dart` 中的 map（`mapZh`, `mapEn`）并保证两个语言映射都包含相应键。
- 版本号在 `pubspec.yaml`（`version:`），平台特定信息在 `android/` 和 `ios/` 对应目录下。修改版本时同时更新这些位置视需要。

## 构建 / 运行 / 调试 快速命令（Windows PowerShell）

```powershell
# 安装依赖
flutter pub get

# 在连接设备或模拟器上运行（自动选择设备）
flutter run

# 指定平台（示例：在 Windows 桌面上运行）
flutter run -d windows

# 构建发行包（Android APK）
flutter build apk --release

# 运行测试
flutter test
```

注意：Android 可以借助 `android/gradlew.bat` 做更底层 Gradle 操作，但通常使用 `flutter build` 即可。

## 代码修改注意事项（AI 代理指引）

- 小步提交：当改动 UI 文本或语言映射时，只修改 `lib/language.dart` 的 map 并运行 `flutter run` 做快速热重载验证。
- 避免引入大型第三方库，除非必要：当前项目依赖极少，新增依赖会改变运行环境。若要添加，请同时更新 `pubspec.yaml` 并运行 `flutter pub get`。
- 主题与颜色：`AppConfig` 中声明的 `primaryColor` / `accentColor` 被 `main.dart` 消费；修改时检查 `ThemeData` 的使用处。

## 常见集成点和跨组件通信

- 本仓库目前未使用平台通道（`MethodChannel`）或后台服务。若需要接入原生功能，应在 `android/` 与 `ios/Runner` 下添加对应实现并在 Dart 端新建桥接层。

## 当你不确定时（优先级）

1. 先查看 `lib/` 下文件，定位 AppConfig / Localize 用法。
2. 本地运行：`flutter pub get` → `flutter run`，验证 UI 与本地化输出。
3. 若改动 `android/` 或 `ios/`，在本机平台（模拟器或真实设备）验证构建。

## 我已检查的地方（便于审阅者快速定位）

- `lib/main.dart` — 应用入口、`AppConfig`、`textLocalize` 使用示例。
- `lib/language.dart` — 语言映射实现（`mapZh`, `mapEn`, `languageMaps`）。
- `pubspec.yaml` — 依赖与版本声明。

如果你希望我把文档改为英文、补充更多构建/CI 细节（例如 GitHub Actions）、或将本地化迁移为 `intl`/arb 流程，请告诉我哪一项优先，我会据此更新该文件。
