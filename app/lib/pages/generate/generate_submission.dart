part of '../generate.dart';

extension _GenerateSubmissionX on _GeneratePageState {
  Map<String, dynamic>? _videoTaskParamsFor(String taskType) {
    switch (taskType) {
      case 'video_dual_chain':
        return {
          'slow_pipeline': 'video_3dgs',
        };
      case 'da3_feed_forward_3dgs':
        return {
          'frame_interval': 5,
          'conf_threshold': 0.5,
        };
      case 'da3_sugar':
        return {
          'regularization': 'dn_consistency',
          'refinement_time': 'short',
          'fast_mode': true,
        };
      case 'da3_2dgs':
        return {
          'iterations': 30000,
          'extract_fps': 2.0,
          'min_images': 24,
        };
      case 'sparse2dgs':
        return {
          'video_sample_count': 12,
          'video_random_seed': 42,
          'min_video_frame_gap': 3,
          'video_max_edge': 0,
        };
      default:
        return null;
    }
  }

  Future<String?> _showImageTaskTypeSheet() {
    final completer = Completer<String?>();
    TDActionSheet(
      context,
      description: textLocalize('gen_sheet_desc'),
      items: [
        TDActionSheetItem(label: textLocalize('gen_sheet_object')),
        TDActionSheetItem(label: textLocalize('gen_sheet_scene')),
      ],
      cancelText: textLocalize("gen_cancel"),
      onSelected: (item, index) {
        completer.complete(
          index == 0 ? 'single_image_sam3d' : 'single_image_sharp',
        );
      },
      onCancel: () {
        if (!completer.isCompleted) completer.complete(null);
      },
      onClose: () {
        if (!completer.isCompleted) completer.complete(null);
      },
    ).show();
    return completer.future;
  }

  void _showTextImagePreview(String prompt) {
    var didConfirm = false;
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (sheetContext) {
        final isDark = Theme.of(sheetContext).brightness == Brightness.dark;
        final panelColor = isDark
            ? AppTheme.darkSurface.withValues(alpha: 0.96)
            : BDDesign.colorPaperWhite.withValues(alpha: 0.98);
        final dividerColor = isDark
            ? Colors.white.withValues(alpha: 0.08)
            : BDDesign.colorMutedBlue.withValues(alpha: 0.10);
        final titleColor = isDark
            ? BDDesign.colorPaperWhite
            : BDDesign.colorInkBlack;
        final hintColor = isDark
            ? Colors.white.withValues(alpha: 0.62)
            : BDDesign.colorMutedBlue;

        return StatefulBuilder(
          builder: (builderContext, setSheetState) {
            return Container(
              height: MediaQuery.of(context).size.height * 0.75,
              decoration: BoxDecoration(
                color: panelColor,
                borderRadius: const BorderRadius.vertical(
                  top: Radius.circular(28),
                ),
                border: Border.all(color: dividerColor),
              ),
              child: Column(
                children: [
                  Padding(
                    padding: const EdgeInsets.fromLTRB(20, 18, 16, 14),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                textLocalize('gen_text_preview_title'),
                                style: TextStyle(
                                  color: titleColor,
                                  fontSize: 19,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                textLocalize('gen_preview_subtitle'),
                                style: TextStyle(
                                  color: hintColor,
                                  fontSize: 12.5,
                                ),
                              ),
                            ],
                          ),
                        ),
                        GestureDetector(
                          onTap: () => Navigator.pop(sheetContext),
                          child: Icon(Icons.close, color: hintColor),
                        ),
                      ],
                    ),
                  ),
                  Divider(height: 1, color: dividerColor),
                  Expanded(
                    child: _isGenerating
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const CircularProgressIndicator(),
                                const SizedBox(height: 16),
                                Text(
                                  textLocalize('gen_text_generating'),
                                  style: TextStyle(color: hintColor),
                                ),
                              ],
                            ),
                          )
                        : _generatedImageUrl != null
                        ? Padding(
                            padding: const EdgeInsets.all(16),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(20),
                              child: Image.network(
                                _generatedImageUrl!,
                                fit: BoxFit.contain,
                                loadingBuilder:
                                    (context, child, loadingProgress) {
                                      if (loadingProgress == null) {
                                        return child;
                                      }
                                      return const Center(
                                        child: CircularProgressIndicator(),
                                      );
                                    },
                                errorBuilder: (context, error, stackTrace) {
                                  return Center(
                                    child: Text(
                                      textLocalize('gen_image_load_fail'),
                                      style: TextStyle(color: hintColor),
                                    ),
                                  );
                                },
                              ),
                            ),
                          )
                        : const SizedBox.shrink(),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
                    child: Row(
                      children: [
                        Expanded(
                          child: TDButton(
                            onTap: _isGenerating
                                ? () {}
                                : () async {
                                    setSheetState(() {});
                                    _refresh(() {
                                      _isGenerating = true;
                                    });
                                    try {
                                      final response = await Supabase
                                          .instance
                                          .client
                                          .functions
                                          .invoke(
                                            'text-to-image',
                                            body: {'prompt': prompt},
                                          );
                                      final data = response.data;
                                      if (data is Map &&
                                          data['success'] == true &&
                                          data['image_url'] != null) {
                                        _refresh(() {
                                          _generatedImageUrl =
                                              data['image_url'] as String;
                                        });
                                      } else if (mounted) {
                                        TDToast.showText(
                                          textLocalize('gen_regenerate_fail'),
                                          context: context,
                                        );
                                      }
                                    } catch (e) {
                                      if (mounted) {
                                        TDToast.showText(
                                          '${textLocalize('gen_regenerate_fail')}: $e',
                                          context: context,
                                        );
                                      }
                                    } finally {
                                      _refresh(() {
                                        _isGenerating = false;
                                      });
                                      setSheetState(() {});
                                    }
                                  },
                            text: textLocalize('gen_text_regenerate'),
                            style: TDButtonStyle(
                              backgroundColor: isDark
                                  ? AppTheme.darkSurfaceElevated
                                  : BDDesign.colorMutedBlueLight,
                              textColor: titleColor,
                              radius: BorderRadius.circular(18),
                            ),
                            theme: TDButtonTheme.defaultTheme,
                            size: TDButtonSize.large,
                            shape: TDButtonShape.rectangle,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: TDButton(
                            onTap: (_isGenerating || _generatedImageUrl == null)
                                ? () {}
                                : () async {
                                    didConfirm = true;
                                    Navigator.pop(sheetContext);
                                    await _confirmTextImage(prompt);
                                  },
                            text: textLocalize('gen_text_confirm'),
                            style: TDButtonStyle(
                              backgroundColor: BDDesign.colorMutedBlue,
                              textColor: Colors.white,
                              radius: BorderRadius.circular(18),
                            ),
                            type: TDButtonType.fill,
                            theme: TDButtonTheme.primary,
                            size: TDButtonSize.large,
                            shape: TDButtonShape.rectangle,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    ).whenComplete(() {
      if (!mounted || didConfirm) {
        return;
      }
      _refresh(() {
        _generatedImageUrl = null;
        _textEditingController.clear();
        GenConfig.uploadedText = '';
      });
    });
  }

  Future<void> _confirmTextImage(String prompt) async {
    final client = Supabase.instance.client;
    var user = client.auth.currentUser;
    if (user == null) {
      if (SupabaseConfig.isAdminMode) {
        if (mounted) {
          TDToast.showText('当前为管理员浏览模式，未绑定用户，暂不支持直接提交生成任务。', context: context);
        }
        return;
      }
      if (mounted) {
        TDToast.showText(textLocalize('not_logged_in'), context: context);
        await Navigator.pushNamed(context, '/login');
      }
      user = client.auth.currentUser;
      if (user == null) {
        if (mounted) TDToast.showText('登录已取消或未完成', context: context);
        return;
      }
    }

    _refresh(() {
      _isUploading = true;
    });

    try {
      final response = await client.functions.invoke(
        'confirm-text-image',
        body: {'image_url': _generatedImageUrl, 'prompt': prompt},
      );

      final data = response.data;
      if (data is Map && data['success'] == true) {
        if (mounted) {
          TDToast.showText(textLocalize('gen_submit_success'), context: context);
          ref.read(pageIndexProvider.notifier).state = 0;
          final nav = Navigator.of(context);
          _generatedImageUrl = null;
          _textEditingController.clear();
          GenConfig.uploadedText = '';
          nav.pushNamed('/tasks');
        }
      } else {
        final errMsg = (data is Map) ? (data['error'] ?? textLocalize('gen_submit_fail')) : textLocalize('gen_server_error');
        throw Exception(errMsg);
      }
    } on FunctionException catch (e) {
      if (mounted) {
        TDToast.showText('${textLocalize('gen_submit_fail')}: ${e.details}', context: context);
      }
    } catch (e) {
      if (mounted) {
        TDToast.showText('${textLocalize('gen_submit_fail')}: $e', context: context);
      }
    } finally {
      if (mounted) {
        _refresh(() {
          _isUploading = false;
        });
      }
    }
  }

  Future<void> _submit() async {
    if (_tabController.index == 0) {
      await _submitImageTask();
      return;
    }
    if (_tabController.index == 1) {
      await _submitTextTask();
      return;
    }
    if (_tabController.index == 2) {
      await _submitVideoTask();
    }
  }

  Future<void> _submitImageTask() async {
    if (GenConfig.uploadedImages.isEmpty) {
      if (mounted) TDToast.showText(textLocalize('gen_select_image'), context: context);
      return;
    }

    final taskType = await _showImageTaskTypeSheet();
    if (taskType == null) {
      return;
    }

    final client = Supabase.instance.client;
    final user = await _requireAuthenticatedUser(
      adminModeMessage: textLocalize('admin_mode_msg'),
    );
    if (user == null) {
      return;
    }

    _refresh(() {
      _isUploading = true;
    });

    try {
      final sceneId = _GeneratePageState._generateSceneId();
      await _uploadAssetToStorage(
        userId: user.id,
        sceneId: sceneId,
        localPath: GenConfig.uploadedImages[0].assetPath!,
        storageFileName: 'image.png',
        contentType: 'image/png',
      );

      await client.from("processing_tasks").insert({
        'scene_id': sceneId,
        'user_id': user.id,
        'status': 'pending',
        'task_type': taskType,
      });

      if (mounted) {
        TDToast.showText(textLocalize('gen_submit_success'), context: context);
        ref.read(pageIndexProvider.notifier).state = 0;
        final nav = Navigator.of(context);
        GenConfig.uploadedImages.clear();
        nav.pushNamed('/tasks');
      }
    } catch (e) {
      if (mounted) {
        TDToast.showText('${textLocalize('gen_submit_fail')}: $e', context: context);
      }
    } finally {
      if (mounted) {
        _refresh(() {
          _isUploading = false;
        });
      }
    }
  }

  Future<void> _submitTextTask() async {
    final prompt = _textEditingController.text.trim();
    if (prompt.isEmpty) {
      if (mounted) TDToast.showText(textLocalize('gen_enter_text'), context: context);
      return;
    }

    _refresh(() {
      _isGenerating = true;
    });

    try {
      final response = await Supabase.instance.client.functions.invoke(
        'text-to-image',
        body: {'prompt': prompt},
      );

      final data = response.data;
      if (data is Map && data['success'] == true && data['image_url'] != null) {
        final imageUrl = data['image_url'] as String;
        _refresh(() {
          _generatedImageUrl = imageUrl;
          _isGenerating = false;
        });
        if (mounted) {
          _showTextImagePreview(prompt);
        }
      } else {
        final errMsg = (data is Map) ? (data['error'] ?? textLocalize('gen_generate_fail')) : textLocalize('gen_server_error');
        throw Exception(errMsg);
      }
    } on FunctionException catch (e) {
      if (mounted) {
        TDToast.showText('${textLocalize('gen_generate_fail')}: ${e.details}', context: context);
      }
    } catch (e) {
      if (mounted) {
        TDToast.showText('${textLocalize('gen_generate_fail')}: $e', context: context);
      }
    } finally {
      if (mounted) {
        _refresh(() {
          _isGenerating = false;
        });
      }
    }
  }

  Future<void> _submitVideoTask() async {
    if (GenConfig.uploadedVideos.isEmpty) {
      if (mounted) TDToast.showText(textLocalize('gen_select_video'), context: context);
      return;
    }

    final client = Supabase.instance.client;
    final user = await _requireAuthenticatedUser(
      adminModeMessage: textLocalize('admin_mode_msg'),
    );
    if (user == null) {
      return;
    }

    _refresh(() {
      _isUploading = true;
    });

    try {
      final sceneId = _GeneratePageState._generateSceneId();
      final taskType = _selectedVideoTaskType;
      final taskParams = _videoTaskParamsFor(taskType);
      await _uploadAssetToStorage(
        userId: user.id,
        sceneId: sceneId,
        localPath: GenConfig.uploadedVideos[0].assetPath!,
        storageFileName: 'video.mp4',
        contentType: 'video/mp4',
      );

      await client.from("processing_tasks").insert({
        'scene_id': sceneId,
        'user_id': user.id,
        'status': 'pending',
        'task_type': taskType,
        if (taskParams != null) 'task_params': taskParams,
      });

      if (mounted) {
        TDToast.showText(textLocalize('gen_submit_success'), context: context);
        ref.read(pageIndexProvider.notifier).state = 0;
        final nav = Navigator.of(context);
        GenConfig.uploadedVideos.clear();
        _selectedVideoTaskType = 'video_3dgs';
        nav.pushNamed('/tasks');
      }
    } catch (e) {
      if (mounted) {
        TDToast.showText('${textLocalize('gen_submit_fail')}: $e', context: context);
      }
    } finally {
      if (mounted) {
        _refresh(() {
          _isUploading = false;
        });
      }
    }
  }

  Future<User?> _requireAuthenticatedUser({
    required String adminModeMessage,
  }) async {
    final client = Supabase.instance.client;
    var user = client.auth.currentUser;
    if (user != null) {
      return user;
    }

    if (SupabaseConfig.isAdminMode) {
      if (mounted) {
        TDToast.showText(adminModeMessage, context: context);
      }
      return null;
    }

    if (mounted) {
      TDToast.showText(textLocalize('not_logged_in'), context: context);
      await Navigator.pushNamed(context, '/login');
    }

    user = client.auth.currentUser;
    if (user == null) {
      if (mounted) {
        TDToast.showText(textLocalize('login_cancelled'), context: context);
      }
      return null;
    }

    if (mounted) {
      TDToast.showText(textLocalize('login_success_upload'), context: context);
    }
    return user;
  }

  Future<void> _uploadAssetToStorage({
    required String userId,
    required String sceneId,
    required String localPath,
    required String storageFileName,
    required String contentType,
  }) async {
    final client = Supabase.instance.client;
    final file = File(localPath);
    final fileSize = await file.length();
    final storagePath = '$userId/$sceneId/raw/$storageFileName';
    final url =
        '${SupabaseConfig.url}/storage/v1/object/braindance-assets/$storagePath';
    final dio = Dio();

    _refresh(() {
      _totalFileSize = fileSize;
      _uploadedBytes = 0;
    });

    await dio.post(
      url,
      data: file.openRead(),
      options: Options(
        headers: {
          'Authorization': 'Bearer ${client.auth.currentSession?.accessToken}',
          'apikey': SupabaseConfig.apiKey,
          'Content-Type': contentType,
          'Content-Length': fileSize.toString(),
        },
      ),
      onSendProgress: (count, total) {
        if (mounted) {
          _refresh(() {
            _uploadedBytes = count;
            _uploadProgress = count / fileSize;
          });
        }
      },
    );
  }
}
