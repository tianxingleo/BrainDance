// agent_recall_models 导出，保持旧调用方 import 路径不变
export 'agent_recall_models.dart';

import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../configs/app_config.dart';
import '../configs/supabase_config.dart';
import 'agent_recall_error.dart';
import 'agent_recall_models.dart';
import 'agent_recall_streaming.dart';

// ── Service ──────────────────────────────────────────────────

class AgentRecallService {
  final SupabaseClient _client = Supabase.instance.client;

  Stream<String> queryStream(
    String query, {
    List<String>? selectedModelIds,
    String executionMode = 'preview',
    String? currentSceneId,
    String? currentModelId,
    String? currentMode,
    List<String>? candidateSceneIds,
    String? sessionId,
    String? conversationSummary,
    AgentSessionState? sessionState,
  }) async* {
    final trimmedQuery = query.trim();
    if (trimmedQuery.isEmpty) {
      throw Exception(textLocalize('agent_error_empty_query'));
    }

    final endpoint = SupabaseConfig.edgeFunctionUrl('agent-recall');
    if (endpoint.isEmpty) {
      throw Exception(textLocalize('agent_error_missing_supabase_url'));
    }

    final dio = Dio();
    // 配置超时 (根据需要调整)
    dio.options.connectTimeout = const Duration(seconds: 20);
    dio.options.receiveTimeout = const Duration(seconds: 300); // 长连接需较长超时

    try {
      final session = _client.auth.currentSession;
      final token = session?.accessToken;
      final apiKey = SupabaseConfig.apiKey;

      final response = await dio.post<ResponseBody>(
        endpoint,
        queryParameters: {'stream': '1'},
        options: Options(
          responseType: ResponseType.stream,
          headers: {
            'Authorization': token != null ? 'Bearer $token' : null,
            'apikey': apiKey,
            'Accept': 'text/event-stream, application/x-ndjson',
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
          },
        ),
        data: {
          'query': trimmedQuery,
          if (selectedModelIds != null) 'selectedModelIds': selectedModelIds,
          'executionMode': executionMode,
          if (currentSceneId != null) 'currentSceneId': currentSceneId,
          if (currentModelId != null) 'currentModelId': currentModelId,
          if (currentMode != null) 'currentMode': currentMode,
          if (candidateSceneIds != null) 'candidateSceneIds': candidateSceneIds,
          if (sessionId != null) 'sessionId': sessionId,
          if (conversationSummary != null)
            'conversationSummary': conversationSummary,
          if (sessionState != null) 'sessionState': sessionState.toJson(),
        },
      );

      final stream = response.data?.stream;
      if (stream == null) {
        throw Exception('Response stream is null');
      }

      final eventBuffer = StringBuffer();
      await for (final chunk in stream.cast<List<int>>().transform(
        utf8.decoder,
      )) {
        if (chunk.isEmpty) {
          continue;
        }
        eventBuffer.write(chunk);
        final rawBuffer = eventBuffer.toString();
        final parsed = drainStreamingEvents(rawBuffer);
        if (parsed.remaining != rawBuffer.length) {
          eventBuffer
            ..clear()
            ..write(rawBuffer.substring(parsed.remaining));
        }
        for (final event in parsed.events) {
          yield event;
        }
      }

      final tail = eventBuffer.toString().trim();
      if (tail.isNotEmpty) {
        for (final event in parseEventChunk(tail)) {
          yield event;
        }
      }
    } catch (e) {
      if (kDebugMode) {
        print('Agent streaming error: $e');
      }

      try {
        final result = await this.query(
          trimmedQuery,
          selectedModelIds: selectedModelIds,
          executionMode: executionMode,
          currentSceneId: currentSceneId,
          currentModelId: currentModelId,
          currentMode: currentMode,
          candidateSceneIds: candidateSceneIds,
          sessionId: sessionId,
          conversationSummary: conversationSummary,
          sessionState: sessionState,
        );
        yield jsonEncode({'event': 'done', 'data': _encodeResponse(result)});
      } catch (fallbackError) {
        yield jsonEncode({
          'event': 'error',
          'data': normalizeInvokeError(fallbackError),
        });
      }
    } finally {
      dio.close();
    }
  }

  Future<AgentRecallResponse> query(
    String query, {
    List<String>? selectedModelIds,
    String executionMode = 'preview',
    String? currentSceneId,
    String? currentModelId,
    String? currentMode,
    List<String>? candidateSceneIds,
    String? sessionId,
    String? conversationSummary,
    AgentSessionState? sessionState,
  }) async {
    final trimmedQuery = query.trim();
    if (trimmedQuery.isEmpty) {
      throw Exception(textLocalize('agent_error_empty_query'));
    }

    try {
      final response = await _client.functions.invoke(
        'agent-recall',
        body: {
          'query': trimmedQuery,
          if (selectedModelIds != null) 'selectedModelIds': selectedModelIds,
          'executionMode': executionMode,
          if (currentSceneId != null) 'currentSceneId': currentSceneId,
          if (currentModelId != null) 'currentModelId': currentModelId,
          if (currentMode != null) 'currentMode': currentMode,
          if (candidateSceneIds != null) 'candidateSceneIds': candidateSceneIds,
          if (sessionId != null) 'sessionId': sessionId,
          if (conversationSummary != null)
            'conversationSummary': conversationSummary,
          if (sessionState != null) 'sessionState': sessionState.toJson(),
        },
      );

      final data = decodeInvokeData(response.data);
      if (data is! Map) {
        throw Exception(textLocalize('agent_error_bad_response'));
      }

      if (data['error'] != null) {
        throw Exception(data['error'].toString());
      }

      return AgentRecallResponse.fromJson(Map<String, dynamic>.from(data));
    } catch (e) {
      throw Exception(normalizeInvokeError(e));
    }
  }

  Map<String, dynamic> _encodeResponse(AgentRecallResponse response) {
    return {
      'mode': response.mode,
      'answer': response.answer,
      'evidence': response.evidence == null
          ? null
          : {
              'sceneId': response.evidence!.sceneId,
              'similarity': response.evidence!.similarity,
              'matchedFrames': response.evidence!.matchedFrames
                  .map(
                    (frame) => {
                      'imageName': frame.imageName,
                      'similarity': frame.similarity,
                      'transformMatrix': frame.transformMatrix,
                    },
                  )
                  .toList(),
            },
      'actions': response.actions
          .map(
            (action) => {
              'type': action.type,
              'payload': {
                'sceneId': action.sceneId,
                if (action.modelId != null) 'modelId': action.modelId,
                if (action.ply != null) 'ply': action.ply,
                if (action.poses != null) 'poses': action.poses,
                if (action.imageName != null) 'imageId': action.imageName,
                if (action.matrix != null) 'matrix': action.matrix,
              },
            },
          )
          .toList(),
      'top_candidates': response.candidates
          .map(
            (candidate) => {
              'scene_id': candidate.sceneId,
              'model_id': candidate.modelId,
              'score': candidate.score,
              'description': candidate.description,
              'pose_image_id': candidate.poseImageId,
            },
          )
          .toList(),
      'tool_trace': response.toolTrace
          .map(
            (trace) => {
              'tool_name': trace.toolName,
              'args': trace.args,
              'result_summary': trace.resultSummary,
            },
          )
          .toList(),
      'response_resolution': response.responseResolution,
      'selected_candidate_reason': response.selectedCandidateReason,
      'asset_context': response.assetContext,
      'compare_context': response.compareContext,
      'collection_context': response.collectionContext,
      'creative_context': response.creativeContext,
      'memory_graph_context': response.memoryGraphContext,
      'session_state': response.sessionState?.toJson(),
      'conversation_summary': response.conversationSummary,
      'follow_up': response.followUp == null
          ? null
          : {
              'status': response.followUp!.status,
              'kind': response.followUp!.kind,
              'message': response.followUp!.message,
              'input_placeholder': response.followUp!.inputPlaceholder,
              'suggested_replies': response.followUp!.suggestedReplies,
            },
    };
  }
}
