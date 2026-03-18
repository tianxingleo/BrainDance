import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

import '../configs/app_theme.dart';
import '../configs/supabase_config.dart';
import '../extra_func/dynamic_background.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLoading = false;
  bool _isSignUp = false; // 切换注册/登录状态

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> _handleAuth() async {
    setState(() {
      _isLoading = true;
    });

    try {
      if (_isSignUp) {
        // 注册流程
        await Supabase.instance.client.auth.signUp(
          email: _emailController.text.trim(),
          password: _passwordController.text.trim(),
        );
        if (mounted) {
          TDToast.showText('注册成功！请检查邮箱完成验证。', context: context);
          setState(() {
            _isSignUp = false; // 注册成功后切回登录界面
          });
        }
      } else {
        // 登录流程
        await Supabase.instance.client.auth.signInWithPassword(
          email: _emailController.text.trim(),
          password: _passwordController.text.trim(),
        );
        if (mounted) {
          TDToast.showSuccess('登录成功', context: context);
          Navigator.of(context).pushReplacementNamed('/'); // 回到首页
        }
      }
    } on AuthException catch (e) {
      if (mounted) {
        TDToast.showText('认证失败: ${e.message}', context: context);
      }
    } catch (e) {
      if (mounted) {
        TDToast.showText('发生错误: $e', context: context);
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = TDTheme.of(context);
    final colorScheme = Theme.of(context).colorScheme;

    if (SupabaseConfig.isAdminMode) {
      return Scaffold(
        backgroundColor: context.appPageBackground,
        appBar: AppBar(title: const Text('Admin Mode')),
        body: DynamicGradientBackground(
          child: Center(
            child: Container(
              constraints: const BoxConstraints(maxWidth: 480),
              margin: const EdgeInsets.all(24.0),
              padding: const EdgeInsets.all(24.0),
              decoration: BoxDecoration(
                color: context.appSurfaceMutedColor,
                borderRadius: BorderRadius.circular(theme.radiusExtraLarge),
                border: Border.all(color: context.appBorderColor),
                boxShadow: context.appCardShadow,
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    Icons.admin_panel_settings,
                    size: 64,
                    color: colorScheme.primary,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    '当前 .env 使用的是 secret key，应用已进入管理员模式。',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: colorScheme.onSurface),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '管理员模式会绕过 RLS，可直接查看全部模型与任务。',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: colorScheme.onSurface.withValues(alpha: 0.72),
                    ),
                  ),
                  const SizedBox(height: 24),
                  TDButton(
                    text: '进入首页',
                    onTap: () {
                      Navigator.of(context).pushReplacementNamed('/');
                    },
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: context.appPageBackground,
      appBar: AppBar(
        title: TDText(
          _isSignUp ? '注册' : '登录',
          font: theme.fontHeadlineSmall,
          fontWeight: FontWeight.w600,
          textColor: colorScheme.onSurface,
        ),
        centerTitle: true,
      ),
      body: DynamicGradientBackground(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: Container(
              constraints: const BoxConstraints(maxWidth: 480),
              padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
              decoration: BoxDecoration(
                color: context.appSurfaceMutedColor,
                borderRadius: BorderRadius.circular(theme.radiusExtraLarge),
                border: Border.all(color: context.appBorderColor),
                boxShadow: context.appCardShadow,
              ),
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 300),
                transitionBuilder: (Widget child, Animation<double> animation) {
                  return FadeTransition(
                    opacity: animation,
                    child: SlideTransition(
                      position: Tween<Offset>(
                        begin: const Offset(0.0, 0.05),
                        end: Offset.zero,
                      ).animate(animation),
                      child: child,
                    ),
                  );
                },
                child: Column(
                  key: ValueKey<bool>(_isSignUp),
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    TDText(
                      _isSignUp ? '创建 Brain Dance 账号' : '登录 Brain Dance',
                      font: theme.fontTitleLarge,
                      fontWeight: FontWeight.w700,
                      textColor: colorScheme.onSurface,
                    ),
                    const SizedBox(height: 8),
                    TDText(
                      _isSignUp ? '注册后请前往邮箱完成验证。' : '输入邮箱和密码继续。',
                      font: theme.fontBodyMedium,
                      textColor: colorScheme.onSurface.withValues(alpha: 0.68),
                    ),
                    const SizedBox(height: 28),
                    TDInput(
                      controller: _emailController,
                      type: TDInputType.normal,
                      leftLabel: '邮箱',
                      hintText: '请输入验证邮箱',
                    ),
                    const SizedBox(height: 16),
                    TDInput(
                      controller: _passwordController,
                      type: TDInputType.normal,
                      obscureText: true,
                      leftLabel: '密码',
                      hintText: '请输入密码(至少6位)',
                    ),
                    const SizedBox(height: 32),
                    _isLoading
                        ? const CircularProgressIndicator()
                        : TDButton(
                            text: _isSignUp ? '注册新账号' : '登录',
                            type: TDButtonType.fill,
                            theme: TDButtonTheme.primary,
                            shape: TDButtonShape.round,
                            size: TDButtonSize.large,
                            isBlock: true,
                            onTap: _handleAuth,
                          ),
                    const SizedBox(height: 16),
                    TDButton(
                      text: _isSignUp ? '已有账号？去登录' : '没有账号？去注册',
                      type: TDButtonType.text,
                      theme: TDButtonTheme.primary,
                      onTap: () {
                        setState(() {
                          _isSignUp = !_isSignUp;
                        });
                      },
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
