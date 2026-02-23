import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../configs/app_config.dart';
import '../extra_func/dynamic_background.dart';
import 'webgl_viewer.dart';

class RecallPage extends StatefulWidget {
  const RecallPage({super.key});

  @override
  State<RecallPage> createState() => _RecallPageState();
}

class _RecallPageState extends State<RecallPage> {
  List<Map<String, dynamic>> _models = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchModels();
  }

  /// 将 Storage 内的相对路径转为可访问的公开 URL。
  /// ply_path 示例: "my_scene/point_cloud.ply"
  String _toPublicUrl(String storagePath) {
    try {
      return Supabase.instance.client.storage
          .from('braindance-assets')
          .getPublicUrl(storagePath);
    } catch (_) {
      return storagePath; // 兜底：原样返回，让 viewer 显示错误提示
    }
  }

  Future<void> _fetchModels() async {
    try {
      final response = await Supabase.instance.client
          .from('model_assets')
          .select('id, scene_id, description, ply_path, preview_img_path, created_at')
          .order('created_at', ascending: false);

      if (mounted) {
        setState(() {
          _models = List<Map<String, dynamic>>.from(response);
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        TDToast.showText('加载模型失败: $e', context: context);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: TDTheme.of(context).grayColor1,
      appBar: AppBar(
        backgroundColor: TDTheme.of(context).whiteColor1.withValues(alpha: 0.95),
        title: Container(
          alignment: Alignment.centerLeft,
          child: TDText(
            textLocalize("home_page"),
            font: TDTheme.of(context).fontHeadlineSmall,
            fontWeight: FontWeight.w600,
            textColor: TDTheme.of(context).fontGyColor1,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.blue),
            tooltip: '刷新',
            onPressed: () {
              setState(() {
                _isLoading = true;
              });
              _fetchModels();
            },
          )
        ],
        toolbarHeight: 60,
        elevation: 0,
      ),
      extendBodyBehindAppBar: true,
      body: DynamicGradientBackground(
        child: _isLoading
            ? const Center(child: CircularProgressIndicator())
            : _models.isEmpty
                ? _buildEmptyState()
                : _buildModelGrid(),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.85,
        padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
        decoration: BoxDecoration(
          color: TDTheme.of(context).whiteColor1.withValues(alpha: 0.8),
          borderRadius: BorderRadius.circular(TDTheme.of(context).radiusExtraLarge),
          border: Border.all(
            color: TDTheme.of(context).whiteColor1,
            width: 1,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.05),
              blurRadius: 20,
              spreadRadius: 5,
            )
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TDImage(
              assetUrl: 'assets/sprites/empty_state.png',
              width: 120,
              height: 120,
              errorWidget: Icon(
                TDIcons.time_filled,
                size: 80,
                color: TDTheme.of(context).brandColor4,
              ),
            ),
            const SizedBox(height: 24),
            TDText(
              textLocalize("home_page"),
              font: TDTheme.of(context).fontTitleLarge,
              textColor: TDTheme.of(context).fontGyColor1,
              fontWeight: FontWeight.w600,
            ),
            const SizedBox(height: 8),
            TDText(
              "暂无回忆，去记录一些美好瞬间吧",
              font: TDTheme.of(context).fontBodyMedium,
              textColor: TDTheme.of(context).fontGyColor3,
            ),
            const SizedBox(height: 40),
            TDButton(
              text: "打开 3D 查看器",
              iconWidget: const Icon(TDIcons.view_module, color: Colors.white, size: 20),
              type: TDButtonType.fill,
              theme: TDButtonTheme.primary,
              shape: TDButtonShape.round,
              size: TDButtonSize.large,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const WebGLViewerPage(
                      sceneId: 'Demo 场景',
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModelGrid() {
    return SafeArea(
      child: GridView.builder(
        padding: const EdgeInsets.all(16.0),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          crossAxisSpacing: 16.0,
          mainAxisSpacing: 16.0,
          childAspectRatio: 0.85,
        ),
        itemCount: _models.length,
        itemBuilder: (context, index) {
          final model = _models[index];
          final sceneId = model['scene_id'] ?? 'Unknown Scene';
          final desc = model['description'] ?? '没有描述信息';

          return GestureDetector(
            onTap: () {
              final plyPath = model['ply_path'] as String? ?? '';
              final modelUrl = plyPath.isNotEmpty
                  ? _toPublicUrl(plyPath)
                  : './models/scene_auto_sync_raw.ply';
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => WebGLViewerPage(
                    initialModelUrl: modelUrl,
                    sceneId: sceneId,
                  ),
                ),
              );
            },
            child: Container(
              decoration: BoxDecoration(
                color: TDTheme.of(context).whiteColor1.withValues(alpha: 0.9),
                borderRadius: BorderRadius.circular(TDTheme.of(context).radiusLarge),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.05),
                    blurRadius: 10,
                    offset: const Offset(0, 4),
                  )
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(
                    child: Container(
                      decoration: BoxDecoration(
                        color: TDTheme.of(context).grayColor3,
                        borderRadius: BorderRadius.vertical(
                          top: Radius.circular(TDTheme.of(context).radiusLarge),
                        ),
                      ),
                      child: Center(
                        child: Icon(
                          Icons.view_in_ar,
                          size: 40,
                          color: TDTheme.of(context).fontGyColor3,
                        ),
                      ),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        TDText(
                          sceneId,
                          font: TDTheme.of(context).fontTitleMedium,
                          fontWeight: FontWeight.w600,
                          maxLines: 1,
                        ),
                        const SizedBox(height: 4),
                        TDText(
                          desc,
                          font: TDTheme.of(context).fontBodySmall,
                          textColor: TDTheme.of(context).fontGyColor3,
                          maxLines: 2,
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
