part of '../generate.dart';

extension _GenerateSubmissionX on _GeneratePageState {
  void _openTaskListAfterSubmit() {
    Navigator.of(context).pushNamed('/tasks');
  }

  Map<String, dynamic>? _videoTaskParamsFor(String taskType) {
    switch (taskType) {
      case 'video_dual_chain':
        return {
          "mapper_type": "da3",
          "slow_pipeline": "video_3dgs",
          "best_frame_sample_count": 8,
          "sam3d_vram_threshold_gb": 25,
        };
      case 'da3_feed_forward_3dgs':
        return {'frame_interval': 5, 'conf_threshold': 0.5};
      case 'da3_sugar':
        return {
          'regularization': 'dn_consistency',
          'refinement_time': 'short',
          'fast_mode': true,
        };
      case 'da3_2dgs':
        return {'iterations': 30000, 'extract_fps': 2.0, 'min_images': 24};
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
              height: MediaQuery.sizeOf(context).height * 0.75,
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
                                        showAppToast(
                                          context,
                                          textLocalize('gen_regenerate_fail'),
                                        );
                                      }
                                    } catch (e) {
                                      if (mounted) {
                                        debugPrint(
                                          '[GenerateSubmission] regenerate error: $e',
                                        );
                                        showAppToast(
                                          context,
                                          textLocalize('gen_regenerate_fail'),
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
          showAppToast(context, '当前为管理员浏览模式，未绑定用户，暂不支持直接提交生成任务。');
        }
        return;
      }
      if (mounted) {
        showAppToast(context, textLocalize('not_logged_in'));
        await Navigator.pushNamed(context, '/login');
      }
      user = client.auth.currentUser;
      if (user == null) {
        if (mounted) showAppToast(context, '登录已取消或未完成');
        return;
      }
    }

    _refresh(() {
      _isUploading = true;
    });

    try {
      final response = await client.functions.invoke(
        'confirm-text-image',
        body: {
          'image_url': _generatedImageUrl,
          'prompt': prompt,
          'display_name': _modelNameController.text.trim(),
        },
      );

      final data = response.data;
      if (data is Map && data['success'] == true) {
        if (mounted) {
          showAppToast(context, textLocalize('gen_submit_success'));
          _generatedImageUrl = null;
          _textEditingController.clear();
          GenConfig.uploadedText = '';
          _openTaskListAfterSubmit();
        }
      } else {
        final errMsg = (data is Map)
            ? (data['error'] ?? textLocalize('gen_submit_fail'))
            : textLocalize('gen_server_error');
        throw Exception(errMsg);
      }
    } on FunctionException catch (e) {
      if (mounted) {
        debugPrint(
          '[GenerateSubmission] submit FunctionException: ${e.details}',
        );
        showAppToast(context, textLocalize('gen_submit_fail'));
      }
    } catch (e) {
      if (mounted) {
        debugPrint('[GenerateSubmission] submit error: $e');
        showAppToast(context, textLocalize('gen_submit_fail'));
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
    final modelName = _modelNameController.text.trim();
    if (modelName.isEmpty) {
      if (mounted) {
        showAppToast(context, textLocalize('gen_model_name_required'));
        _modelNameFocusNode.requestFocus();
      }
      return;
    }
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
      if (mounted) showAppToast(context, textLocalize('gen_select_image'));
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

    _cancelToken = CancelToken();
    final sceneId = _GeneratePageState._generateSceneId();

    _refresh(() {
      _isUploading = true;
    });

    try {
      await _uploadAssetToStorage(
        userId: user.id,
        sceneId: sceneId,
        localPath: GenConfig.uploadedImages[0].assetPath!,
        storageFileName: 'image.png',
        contentType: 'image/png',
        cancelToken: _cancelToken!,
      );

      await client.from("processing_tasks").insert({
        'scene_id': sceneId,
        'user_id': user.id,
        'status': 'pending',
        'task_type': taskType,
        'display_name': _modelNameController.text.trim(),
      });

      if (mounted) {
        showAppToast(context, textLocalize('gen_submit_success'));
        GenConfig.uploadedImages.clear();
        _openTaskListAfterSubmit();
      }
    } on DioException catch (e) {
      if (e.type == DioExceptionType.cancel) {
        unawaited(
          _deleteStorageAsset(
            userId: user.id,
            sceneId: sceneId,
            fileName: 'image.png',
          ),
        );
        return;
      }
      if (mounted) {
        debugPrint('[GenerateSubmission] image upload error: $e');
        showAppToast(context, textLocalize('gen_submit_fail'));
      }
    } catch (e) {
      if (mounted) {
        debugPrint('[GenerateSubmission] image submit error: $e');
        showAppToast(context, textLocalize('gen_submit_fail'));
      }
    } finally {
      _cancelToken = null;
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
      if (mounted) showAppToast(context, textLocalize('gen_enter_text'));
      return;
    }

    FocusManager.instance.primaryFocus?.unfocus();

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
        final errMsg = (data is Map)
            ? (data['error'] ?? textLocalize('gen_generate_fail'))
            : textLocalize('gen_server_error');
        throw Exception(errMsg);
      }
    } on FunctionException catch (e) {
      if (mounted) {
        debugPrint(
          '[GenerateSubmission] generate FunctionException: ${e.details}',
        );
        showAppToast(context, textLocalize('gen_generate_fail'));
      }
    } catch (e) {
      if (mounted) {
        debugPrint('[GenerateSubmission] generate error: $e');
        showAppToast(context, textLocalize('gen_generate_fail'));
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
      if (mounted) showAppToast(context, textLocalize('gen_select_video'));
      return;
    }

    final client = Supabase.instance.client;
    final user = await _requireAuthenticatedUser(
      adminModeMessage: textLocalize('admin_mode_msg'),
    );
    if (user == null) {
      return;
    }

    _cancelToken = CancelToken();
    final sceneId = _GeneratePageState._generateSceneId();

    _refresh(() {
      _isUploading = true;
    });

    try {
      final taskType = _selectedVideoTaskType;
      final taskParams = _videoTaskParamsFor(taskType);
      await _uploadAssetToStorage(
        userId: user.id,
        sceneId: sceneId,
        localPath: GenConfig.uploadedVideos[0].assetPath!,
        storageFileName: 'video.mp4',
        contentType: 'video/mp4',
        cancelToken: _cancelToken!,
      );

      await client.from("processing_tasks").insert({
        'scene_id': sceneId,
        'user_id': user.id,
        'status': 'pending',
        'task_type': taskType,
        'display_name': _modelNameController.text.trim(),
        if (taskParams != null) 'task_params': taskParams,
      });

      if (mounted) {
        showAppToast(context, textLocalize('gen_submit_success'));
        GenConfig.uploadedVideos.clear();
        _selectedVideoTaskType = 'video_3dgs';
        _openTaskListAfterSubmit();
      }
    } on DioException catch (e) {
      if (e.type == DioExceptionType.cancel) {
        unawaited(
          _deleteStorageAsset(
            userId: user.id,
            sceneId: sceneId,
            fileName: 'video.mp4',
          ),
        );
        return;
      }
      if (mounted) {
        debugPrint('[GenerateSubmission] video upload error: $e');
        showAppToast(context, textLocalize('gen_submit_fail'));
      }
    } catch (e) {
      if (mounted) {
        debugPrint('[GenerateSubmission] video submit error: $e');
        showAppToast(context, textLocalize('gen_submit_fail'));
      }
    } finally {
      _cancelToken = null;
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
        showAppToast(context, adminModeMessage);
      }
      return null;
    }

    if (mounted) {
      showAppToast(context, textLocalize('not_logged_in'));
      await Navigator.pushNamed(context, '/login');
    }

    user = client.auth.currentUser;
    if (user == null) {
      if (mounted) {
        showAppToast(context, textLocalize('login_cancelled'));
      }
      return null;
    }

    if (mounted) {
      showAppToast(context, textLocalize('login_success_upload'));
    }
    return user;
  }

  Future<void> _deleteStorageAsset({
    required String userId,
    required String sceneId,
    required String fileName,
  }) async {
    try {
      await Supabase.instance.client.storage.from('braindance-assets').remove([
        '$userId/$sceneId/raw/$fileName',
      ]);
    } catch (_) {
      // Best-effort cleanup; file may not exist yet on the server.
    }
  }

  Future<void> _uploadAssetToStorage({
    required String userId,
    required String sceneId,
    required String localPath,
    required String storageFileName,
    required String contentType,
    required CancelToken cancelToken,
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
      cancelToken: cancelToken,
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
