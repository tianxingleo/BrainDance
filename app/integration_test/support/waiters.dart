import 'dart:async';

Future<void> waitUntil(
  FutureOr<bool> Function() predicate, {
  Duration timeout = const Duration(seconds: 10),
  Duration interval = const Duration(milliseconds: 250),
}) async {
  final endAt = DateTime.now().add(timeout);
  while (DateTime.now().isBefore(endAt)) {
    if (await predicate()) {
      return;
    }
    await Future<void>.delayed(interval);
  }
  throw TimeoutException('等待条件成立超时: $timeout');
}
