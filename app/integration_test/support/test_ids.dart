class TestIds {
  static const runPrefix = 'it_20260327';

  static String scene(String suffix) => '${runPrefix}_scene_$suffix';
  static String displayName(String suffix) => 'IT-$suffix';
  static String collection(String suffix) => '${runPrefix}_collection_$suffix';
}
