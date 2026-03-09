import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../models/time_peeling_models.dart';
import 'webgl_viewer.dart';

class TimePeelingPage extends StatefulWidget {
  final String? initialSpaceId;
  final String? preferredCaptureId;

  const TimePeelingPage({
    super.key,
    this.initialSpaceId,
    this.preferredCaptureId,
  });

  @override
  State<TimePeelingPage> createState() => _TimePeelingPageState();
}

class _TimePeelingPageState extends State<TimePeelingPage> {
  bool _loading = true;
  List<Map<String, dynamic>> _spaces = [];
  String? _selectedSpaceId;
  List<SpaceCapture> _captures = [];
  SpaceCapture? _base;
  SpaceCapture? _overlay;

  @override
  void initState() {
    super.initState();
    _selectedSpaceId = widget.initialSpaceId;
    _loadSpaces();
  }

  Future<void> _loadSpaces() async {
    setState(() => _loading = true);
    try {
      final user = Supabase.instance.client.auth.currentUser;
      if (user == null) throw Exception('未登录');

      final rows = await Supabase.instance.client
          .from('memory_spaces')
          .select('id, title, created_at')
          .eq('user_id', user.id)
          .order('created_at', ascending: false);

      _spaces = List<Map<String, dynamic>>.from(rows);
      if (_selectedSpaceId == null && _spaces.isNotEmpty) {
        _selectedSpaceId = _spaces.first['id']?.toString();
      }

      if (_selectedSpaceId != null) {
        await _loadCaptures(_selectedSpaceId!);
      }
    } catch (e) {
      if (mounted) {
        TDToast.showText('加载空间失败: $e', context: context);
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _loadCaptures(String spaceId) async {
    final result = await Supabase.instance.client.rpc(
      'get_space_captures',
      params: {'p_space_id': spaceId},
    );

    final captures = List<Map<String, dynamic>>.from(result)
        .map(SpaceCapture.fromJson)
        .toList();

    setState(() {
      _captures = captures;
      _base = captures.isNotEmpty ? captures.first : null;
      _overlay = captures.length > 1 ? captures[1] : (captures.isNotEmpty ? captures.first : null);

      if (widget.preferredCaptureId != null) {
        final preferred = captures.where((c) => c.id == widget.preferredCaptureId).toList();
        if (preferred.isNotEmpty) _overlay = preferred.first;
      }
    });
  }

  Future<void> _openTimePeelingViewer() async {
    if (_selectedSpaceId == null || _base == null || _overlay == null) {
      TDToast.showText('请选择空间和两个时间切片', context: context);
      return;
    }

    try {
      final response = await Supabase.instance.client.functions.invoke(
        'time-peeling-view',
        body: {
          'space_id': _selectedSpaceId,
          'left_capture_id': _base!.id,
          'right_capture_id': _overlay!.id,
        },
      );

      final data = response.data;
      if (data is! Map || data['success'] != true) {
        throw Exception((data is Map ? data['error'] : '请求失败').toString());
      }

      final payload = TimePeelingPayload.fromJson(Map<String, dynamic>.from(data));
      if (!mounted) return;

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => WebGLViewerPage(
            sceneId: 'Time Peeling',
            initialModelUrl: payload.baseModelUrl,
            initialPose: payload.pose,
            timePeelingPayload: payload,
          ),
        ),
      );
    } catch (e) {
      if (mounted) {
        TDToast.showText('打开时光剥离失败: $e', context: context);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('时光剥离')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  DropdownButtonFormField<String>(
                    value: _selectedSpaceId,
                    decoration: const InputDecoration(labelText: '选择空间'),
                    items: _spaces
                        .map((s) => DropdownMenuItem<String>(
                              value: s['id']?.toString(),
                              child: Text((s['title']?.toString().isNotEmpty ?? false)
                                  ? s['title'].toString()
                                  : s['id'].toString()),
                            ))
                        .toList(),
                    onChanged: (v) {
                      if (v == null) return;
                      setState(() => _selectedSpaceId = v);
                      _loadCaptures(v);
                    },
                  ),
                  const SizedBox(height: 16),
                  if (_captures.isEmpty)
                    const Text('该空间暂无切片')
                  else ...[
                    DropdownButtonFormField<String>(
                      value: _base?.id,
                      decoration: const InputDecoration(labelText: '基准切片（当前）'),
                      items: _captures
                          .map((c) => DropdownMenuItem<String>(
                                value: c.id,
                                child: Text('${c.sceneId} | ${c.capturedAt.toLocal()}'),
                              ))
                          .toList(),
                      onChanged: (v) {
                        setState(() {
                          _base = _captures.firstWhere((c) => c.id == v);
                        });
                      },
                    ),
                    const SizedBox(height: 12),
                    DropdownButtonFormField<String>(
                      value: _overlay?.id,
                      decoration: const InputDecoration(labelText: '叠加切片（历史）'),
                      items: _captures
                          .map((c) => DropdownMenuItem<String>(
                                value: c.id,
                                child: Text('${c.sceneId} | ${c.capturedAt.toLocal()} | ${c.status}'),
                              ))
                          .toList(),
                      onChanged: (v) {
                        setState(() {
                          _overlay = _captures.firstWhere((c) => c.id == v);
                        });
                      },
                    ),
                    const SizedBox(height: 24),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: _openTimePeelingViewer,
                        child: const Text('进入时光剥离查看器'),
                      ),
                    ),
                  ],
                ],
              ),
            ),
    );
  }
}
