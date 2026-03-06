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
  final TextEditingController _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _fetchModels();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
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

  /// 根据 PLY 路径推导同场景的 webgl_poses.json 公开 URL。
  /// ply_path 格式：{user_id}/{scene_id}/output/point_cloud.ply
  /// poses 路径：{user_id}/{scene_id}/output/webgl_poses.json
  String? _toPosesUrl(String? plyPath) {
    if (plyPath == null || plyPath.isEmpty) return null;
    try {
      // 将 point_cloud.ply 替换为 webgl_poses.json
      final posesPath = plyPath.replaceAll(RegExp(r'point_cloud\.ply$'), 'webgl_poses.json');
      if (posesPath == plyPath) return null; // 替换失败，路径格式不符
      return Supabase.instance.client.storage
          .from('braindance-assets')
          .getPublicUrl(posesPath);
    } catch (_) {
      return null;
    }
  }

  Future<void> _fetchModels() async {
    try {
      final response = await Supabase.instance.client
          .from('model_assets')
          .select(
            'id, scene_id, description, ply_path, preview_img_path, created_at',
          )
          .order('created_at', ascending: false);

      if (mounted) {
        setState(() {
          _models = List<Map<String, dynamic>>.from(response);
          if (_models.isEmpty) {
            _models.add({
              'id': 'local_demo',
              'scene_id': '本地 Demo 模型 (离线可用)',
              'description': '预置的 3DGS 模型，无需网络即可查看。',
              'ply_path': '',
            });
          }
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _models = [
            {
              'id': 'local_demo',
              'scene_id': '本地 Demo 模型 (离线可用)',
              'description': '预置的 3DGS 模型，无需网络即可查看。',
              'ply_path': '',
            },
          ];
          _isLoading = false;
        });
        TDToast.showText('加载模型失败，已切换至离线模式', context: context);
      }
    }
  }

  Future<void> _searchModels(String query) async {
    if (query.trim().isEmpty) {
      setState(() {
        _isLoading = true;
      });
      _fetchModels();
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final response = await Supabase.instance.client.functions.invoke(
        'search-models',
        body: {'query': query},
      );

      if (response.status == 200) {
        final data = response.data;
        if (data['success'] == true) {
          if (mounted) {
            setState(() {
              _models = List<Map<String, dynamic>>.from(data['results']);
              _isLoading = false;
            });
          }
        } else {
          throw Exception('Search failed: ${data['error']}');
        }
      } else {
        throw Exception('HTTP Error: ${response.status}');
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        TDToast.showText('搜索失败: $e', context: context);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: TDTheme.of(context).grayColor1,
      appBar: AppBar(
        backgroundColor: TDTheme.of(
          context,
        ).whiteColor1.withValues(alpha: 0.95),
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
          ),
        ],
        toolbarHeight: 60,
        elevation: 0,
      ),
      extendBodyBehindAppBar: true,
      body: DynamicGradientBackground(
        child: Column(
          children: [
            SizedBox(
              height: MediaQuery.of(context).padding.top + 60,
            ), // 为 AppBar 留出空间
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: 16.0,
                vertical: 8.0,
              ),
              child: TDSearchBar(
                controller: _searchController,
                placeHolder: '搜索回忆...',
                onSubmitted: (value) {
                  _searchModels(value);
                },
                onTextChanged: (value) {
                  if (value.isEmpty) {
                    _searchModels('');
                  }
                },
              ),
            ),
            Expanded(
              child: _isLoading
                  ? const Center(child: CircularProgressIndicator())
                  : _models.isEmpty
                  ? _buildEmptyState()
                  : _buildModelGrid(),
            ),
          ],
        ),
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
          borderRadius: BorderRadius.circular(
            TDTheme.of(context).radiusExtraLarge,
          ),
          border: Border.all(color: TDTheme.of(context).whiteColor1, width: 1),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.05),
              blurRadius: 20,
              spreadRadius: 5,
            ),
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
              text: "打开本地离线 Demo 模型",
              iconWidget: const Icon(
                TDIcons.view_module,
                color: Colors.white,
                size: 20,
              ),
              type: TDButtonType.fill,
              theme: TDButtonTheme.primary,
              shape: TDButtonShape.round,
              size: TDButtonSize.large,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) =>
                        const WebGLViewerPage(sceneId: '本地 Demo 模型 (离线可用)'),
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
    // If it's search results and has matched_frames, use ListView
    bool isSearchWithFrames = _models.isNotEmpty && _models.first.containsKey('matched_frames');

    if (isSearchWithFrames) {
      return ListView.builder(
        padding: const EdgeInsets.all(16.0),
        itemCount: _models.length,
        itemBuilder: (context, index) {
          final model = _models[index];
          final sceneId = model['scene_id'] ?? 'Unknown Scene';
          final desc = model['description'] ?? '没有描述信息';
          final similarity = model['similarity'] as double?;
          final userId = model['user_id'] ?? '';
          final matchedFrames = model['matched_frames'] as List<dynamic>? ?? [];

          return Container(
            margin: const EdgeInsets.only(bottom: 16.0),
            decoration: BoxDecoration(
              color: TDTheme.of(context).whiteColor1.withValues(alpha: 0.9),
              borderRadius: BorderRadius.circular(TDTheme.of(context).radiusLarge),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.05),
                  blurRadius: 10,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Top header: Model Info
                GestureDetector(
                  onTap: () {
                    _navigateToViewer(model, null);
                  },
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              TDText(sceneId, font: TDTheme.of(context).fontTitleMedium, fontWeight: FontWeight.w600, maxLines: 1),
                              const SizedBox(height: 4),
                              TDText(desc, font: TDTheme.of(context).fontBodySmall, textColor: TDTheme.of(context).fontGyColor3, maxLines: 2),
                            ],
                          ),
                        ),
                        if (similarity != null)
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(color: TDTheme.of(context).brandColor4.withValues(alpha: 0.9), borderRadius: BorderRadius.circular(6)),
                            child: TDText('${(similarity * 100).toStringAsFixed(1)}%', font: TDTheme.of(context).fontBodySmall, textColor: Colors.white),
                          ),
                      ],
                    ),
                  ),
                ),
                // Horizontal list of frames
                if (matchedFrames.isNotEmpty)
                  SizedBox(
                    height: 120,
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      padding: const EdgeInsets.symmetric(horizontal: 16.0).copyWith(bottom: 16.0),
                      itemCount: matchedFrames.length,
                      itemBuilder: (context, frameIndex) {
                        final frame = matchedFrames[frameIndex];
                        final imageName = frame['image_name'];
                        final transformMatrix = frame['transform_matrix'];
                        final frameSim = frame['similarity'] as double?;
                        
                        final imageUrl = "https://kntcynswgrmgbbgntkiv.supabase.co/storage/v1/object/public/braindance-assets/$userId/$sceneId/output/images/$imageName";

                        return GestureDetector(
                          onTap: () {
                            _navigateToViewer(model, transformMatrix);
                          },
                          child: Container(
                            width: 140,
                            margin: const EdgeInsets.only(right: 12.0),
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(8.0),
                              color: TDTheme.of(context).grayColor3,
                              image: DecorationImage(
                                image: NetworkImage(imageUrl),
                                fit: BoxFit.cover,
                              ),
                            ),
                            child: frameSim != null ? Stack(
                              children: [
                                Positioned(
                                  bottom: 4,
                                  left: 4,
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                                    decoration: BoxDecoration(
                                      color: Colors.black.withValues(alpha: 0.6),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                    child: Text(
                                      '${(frameSim * 100).toStringAsFixed(1)}%', 
                                      style: const TextStyle(color: Colors.white, fontSize: 10),
                                    ),
                                  ),
                                ),
                              ],
                            ) : null,
                          ),
                        );
                      },
                    ),
                  ),
              ],
            ),
          );
        },
      );
    }

    return GridView.builder(
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
        final similarity = model['similarity'] as double?;

        return TweenAnimationBuilder<double>(
          tween: Tween(begin: 0.0, end: 1.0),
          duration: Duration(milliseconds: 400 + (index * 100).clamp(0, 600)),
          curve: Curves.easeOutCubic,
          builder: (context, value, child) {
            return Transform.translate(
              offset: Offset(0, 50 * (1 - value)),
              child: Opacity(opacity: value, child: child),
            );
          },
          child: GestureDetector(
            onTap: () {
              _navigateToViewer(model, null);
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
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Container(
                          decoration: BoxDecoration(
                            color: TDTheme.of(context).grayColor3,
                            borderRadius: BorderRadius.vertical(top: Radius.circular(TDTheme.of(context).radiusLarge)),
                          ),
                          child: Center(
                            child: Icon(Icons.view_in_ar, size: 40, color: TDTheme.of(context).fontGyColor3),
                          ),
                        ),
                        if (similarity != null)
                          Positioned(
                            top: 8,
                            right: 8,
                            child: Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                              decoration: BoxDecoration(color: TDTheme.of(context).brandColor4.withValues(alpha: 0.9), borderRadius: BorderRadius.circular(4)),
                              child: TDText('${(similarity * 100).toStringAsFixed(1)}%', font: TDTheme.of(context).fontBodyExtraSmall, textColor: Colors.white),
                            ),
                          ),
                      ],
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        TDText(sceneId, font: TDTheme.of(context).fontTitleMedium, fontWeight: FontWeight.w600, maxLines: 1),
                        const SizedBox(height: 4),
                        TDText(desc, font: TDTheme.of(context).fontBodySmall, textColor: TDTheme.of(context).fontGyColor3, maxLines: 2),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  void _navigateToViewer(Map<String, dynamic> model, dynamic transformMatrix) {
    final plyPath = model['ply_path'] as String? ?? '';
    final modelUrl = plyPath.isNotEmpty ? _toPublicUrl(plyPath) : './models/scene_auto_sync_raw.ply';
    final posesUrl = plyPath.isNotEmpty ? _toPosesUrl(plyPath) : null;
    final sceneId = model['scene_id'] ?? 'Unknown Scene';
    
    // Convert transformMatrix if not null to List<double>
    List<double>? initialPose;
    if (transformMatrix != null && transformMatrix is List) {
      initialPose = transformMatrix.map((e) => (e as num).toDouble()).toList();
    }

    Navigator.push(
      context,
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) => WebGLViewerPage(
          initialModelUrl: modelUrl,
          posesUrl: posesUrl,
          sceneId: sceneId,
          initialPose: initialPose,
        ),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          return FadeTransition(
            opacity: animation,
            child: ScaleTransition(
              scale: Tween<double>(begin: 0.95, end: 1.0).animate(
                CurvedAnimation(parent: animation, curve: Curves.easeOutCubic),
              ),
              child: child,
            ),
          );
        },
      ),
    );
  }
}
