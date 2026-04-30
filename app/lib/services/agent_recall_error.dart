import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../configs/app_config.dart';

// ── 错误处理工具 ──────────────────────────────────────────────

/// 解码 Supabase Edge Function 返回的 data 字段。
///
/// 若为 JSON 字符串则尝试解析，否则原样返回。
Object? decodeInvokeData(Object? data) {
  if (data is String) {
    final trimmed = data.trim();
    if (trimmed.isEmpty) {
      return null;
    }

    try {
      return jsonDecode(trimmed);
    } catch (_) {
      return trimmed;
    }
  }

  return data;
}

/// 将各种异常统一转换为面向用户的错误文本。
String normalizeInvokeError(Object error) {
  if (error is DioException) {
    return _normalizeDioError(error);
  }

  if (error is FunctionException) {
    final detail = decodeInvokeData(error.details);
    if (detail is Map && detail['error'] != null) {
      return _normalizeRawErrorText(detail['error'].toString());
    }
    if (detail is Map && detail['message'] != null) {
      return _normalizeRawErrorText(detail['message'].toString());
    }
    if (detail is String && detail.isNotEmpty) {
      return _normalizeRawErrorText(detail);
    }
    if (error.reasonPhrase != null && error.reasonPhrase!.isNotEmpty) {
      return _normalizeRawErrorText(error.reasonPhrase!);
    }
    if (error.status == 503) {
      return textLocalize('agent_error_unavailable');
    }
    if (error.status == 502 || error.status == 504) {
      return textLocalize('agent_error_upstream');
    }
    return textLocalize(
      'agent_error_http',
    ).replaceAll('{status}', error.status.toString());
  }

  return _normalizeRawErrorText(error.toString());
}

String _normalizeDioError(DioException error) {
  final status = error.response?.statusCode;
  final data = decodeInvokeData(error.response?.data);
  if (data is Map && data['error'] != null) {
    return _normalizeRawErrorText(data['error'].toString());
  }
  if (data is Map && data['message'] != null) {
    return _normalizeRawErrorText(data['message'].toString());
  }
  if (data is String && data.trim().isNotEmpty) {
    return _normalizeRawErrorText(data.trim());
  }
  if (status == 503) {
    return textLocalize('agent_error_unavailable');
  }
  if (status == 502 || status == 504) {
    return textLocalize('agent_error_upstream');
  }
  if (status != null) {
    return textLocalize(
      'agent_error_http',
    ).replaceAll('{status}', status.toString());
  }
  return error.message ?? error.toString();
}

String _normalizeRawErrorText(String message) {
  final trimmed = message.trim();
  if (trimmed.isEmpty) {
    return trimmed;
  }

  final normalized = trimmed.toLowerCase();
  if (normalized.contains(
    'an invalid response was received from the upstream server',
  )) {
    return textLocalize('agent_error_upstream');
  }
  if (normalized.contains('upstream connect error')) {
    return textLocalize('agent_error_upstream');
  }

  return trimmed;
}
