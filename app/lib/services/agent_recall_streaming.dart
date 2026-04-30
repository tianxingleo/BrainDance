import 'dart:convert';

// ── SSE / NDJSON 流解析工具 ──────────────────────────────────

class ParsedStreamEvents {
  final List<String> events;
  final int remaining;

  const ParsedStreamEvents({required this.events, required this.remaining});
}

/// 从原始缓冲区中逐步提取完整的 SSE 或 NDJSON 事件。
///
/// 返回已提取的事件列表以及未消费的剩余偏移量。
ParsedStreamEvents drainStreamingEvents(String raw) {
  final events = <String>[];
  var cursor = 0;

  while (cursor < raw.length) {
    final sseBoundary = raw.indexOf('\n\n', cursor);
    final lineBoundary = raw.indexOf('\n', cursor);
    final looksLikeSse =
        raw.startsWith('event:', cursor) || raw.startsWith('data:', cursor);

    if (looksLikeSse) {
      if (sseBoundary == -1) {
        break;
      }
      final chunk = raw.substring(cursor, sseBoundary).trim();
      cursor = sseBoundary + 2;
      events.addAll(parseEventChunk(chunk));
      continue;
    }

    if (lineBoundary == -1) {
      break;
    }

    final chunk = raw.substring(cursor, lineBoundary).trim();
    cursor = lineBoundary + 1;
    if (chunk.isEmpty) {
      continue;
    }
    events.addAll(parseEventChunk(chunk));
  }

  return ParsedStreamEvents(events: events, remaining: cursor);
}

/// 解析单个 SSE 或 NDJSON chunk，输出标准化 JSON 字符串列表。
List<String> parseEventChunk(String chunk) {
  final trimmed = chunk.trim();
  if (trimmed.isEmpty) {
    return const [];
  }

  if (trimmed.startsWith('{')) {
    return [trimmed];
  }

  String eventName = 'message';
  final dataLines = <String>[];
  for (final line in const LineSplitter().convert(trimmed)) {
    final normalized = line.trimRight();
    if (normalized.isEmpty || normalized.startsWith(':')) {
      continue;
    }
    if (normalized.startsWith('event:')) {
      eventName = normalized.substring(6).trim();
      continue;
    }
    if (normalized.startsWith('data:')) {
      dataLines.add(normalized.substring(5).trimLeft());
    }
  }

  if (dataLines.isEmpty) {
    return const [];
  }

  final dataText = dataLines.join('\n').trim();
  if (dataText.isEmpty) {
    return const [];
  }

  try {
    final decoded = jsonDecode(dataText);
    return [
      jsonEncode({'event': eventName, 'data': decoded}),
    ];
  } catch (_) {
    return [
      jsonEncode({'event': eventName, 'data': dataText}),
    ];
  }
}
