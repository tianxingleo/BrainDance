import 'dart:math';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:braindance/configs/gen_config.dart';
class DebugVar {
  static String taskId = '';
  static VoidCallback? onUpdate;
  static String debugText = '';
}
class UserProfile {
  UserProfile(this.email, this.password);
  String email;
  String password;
}
class DateFormat {
  static String format(int number, int length) {
    return number.toString().padLeft(length, '0');
  }
}
class SupabaseApi {
  static final Random rdg = Random();
  static late final SupabaseClient supabase;
  
  static final UserProfile up = UserProfile("","");
  static final String apiPath = '';
  static final String anonKey = '';
  
  static Future<AuthResponse> signIn(UserProfile up) async {
    final auth = await supabase.auth.signInWithPassword(
      email: up.email,
      password: up.password
    );//可能登录失败！
    return auth;
  }
  static String generateSceneId() {
    DateTime time = DateTime.now();//scene_年月日_6位随机数
    return 'scene_'
    '${DateFormat.format(time.year, 4)}'
    '${DateFormat.format(time.month, 2)}'
    '${DateFormat.format(time.day, 2)}'
    '_'
    '${DateFormat.format(rdg.nextInt(1000000), 6)}';
  }
  static Future<Map<String, dynamic>> createTask(String sceneId) async {
    final res = await supabase.from("processing_tasks").insert({
      'scene_id' : sceneId,
      'user_id' : supabase.auth.currentUser!.id,
      'status' : 'pending',
    }).select();//可能创建失败！
    return res[0];
  }
  static Future<String> uploadVideo(String sceneId) async {
    final path = '${supabase.auth.currentUser!.id}/'
    '$sceneId/raw/video.mp4';
    final result = await Supabase.instance.client.storage.from('braindance-assets').upload(
      path, File(GenConfig.uploadedVideos[0].assetPath!),
    );
    return result;//这是完整路径，包含bucket
  }
  static Future<void> test() async {//是否要做令牌过期处理？
    if (supabase.auth.currentUser == null) {
      DebugVar.debugText = "正在尝试登录...";
      DebugVar.onUpdate?.call();
      await signIn(up);//若没有数据则尝试登录
    }
    final sceneId = generateSceneId();
    DebugVar.debugText = '开始上传视频...';
    DebugVar.onUpdate?.call();
    await uploadVideo(sceneId);
    DebugVar.debugText = '视频上传完成\n'
    '开始创建任务...';
    DebugVar.onUpdate?.call();
    await createTask(sceneId);
    DebugVar.debugText = '任务创建完成';
    DebugVar.onUpdate?.call();
  }
}