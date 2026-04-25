import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'package:braindance/configs/app_config.dart';

import '../configs/app_theme.dart';
import '../configs/supabase_config.dart';
import '../extra_func/dynamic_background.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  Widget _buildSupabaseWarning(
    String message,
    ColorScheme colorScheme,
  ) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: colorScheme.errorContainer.withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        message,
        style: TextStyle(color: colorScheme.onErrorContainer, height: 1.4),
      ),
    );
  }

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
          TDToast.showText(textLocalize('login_signup_success'), context: context);
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
          TDToast.showSuccess(textLocalize('login_success'), context: context);
          Navigator.of(context).pushReplacementNamed('/'); // 回到首页
        }
      }
    } on AuthException catch (e) {
      if (mounted) {
        debugPrint('[Login] auth error: $e');
        TDToast.showText(textLocalize('login_auth_fail'), context: context);
      }
    } catch (e) {
      if (mounted) {
        debugPrint('[Login] error: $e');
        TDToast.showText(textLocalize('login_error'), context: context);
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
    final supabaseDiagnostic = SupabaseConfig.runtimeDiagnosticMessage;

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
                  if (supabaseDiagnostic?.isNotEmpty ?? false)
                    _buildSupabaseWarning(supabaseDiagnostic!, colorScheme),
                  Icon(
                    Icons.admin_panel_settings,
                    size: 64,
                    color: colorScheme.primary,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    textLocalize('login_admin_secret'),
                    textAlign: TextAlign.center,
                    style: TextStyle(color: colorScheme.onSurface),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    textLocalize('login_admin_rls'),
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: colorScheme.onSurface.withValues(alpha: 0.72),
                    ),
                  ),
                  const SizedBox(height: 24),
                  TDButton(
                    text: textLocalize('login_enter_home'),
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
          _isSignUp ? textLocalize('login_signup') : textLocalize('login_login'),
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
                    if (supabaseDiagnostic?.isNotEmpty ?? false)
                      _buildSupabaseWarning(supabaseDiagnostic!, colorScheme),
                    TDText(
                      _isSignUp ? textLocalize('login_create_account') : textLocalize('login_welcome'),
                      font: theme.fontTitleLarge,
                      fontWeight: FontWeight.w700,
                      textColor: colorScheme.onSurface,
                    ),
                    const SizedBox(height: 8),
                    TDText(
                      _isSignUp ? textLocalize('login_signup_hint') : textLocalize('login_hint'),
                      font: theme.fontBodyMedium,
                      textColor: colorScheme.onSurface.withValues(alpha: 0.68),
                    ),
                    const SizedBox(height: 28),
                    TDInput(
                      controller: _emailController,
                      type: TDInputType.normal,
                      leftLabel: textLocalize('login_email'),
                      hintText: textLocalize('login_email_hint'),
                    ),
                    const SizedBox(height: 16),
                    TDInput(
                      controller: _passwordController,
                      type: TDInputType.normal,
                      obscureText: true,
                      leftLabel: textLocalize('login_password'),
                      hintText: textLocalize('login_password_hint'),
                    ),
                    const SizedBox(height: 32),
                    _isLoading
                        ? const CircularProgressIndicator()
                        : TDButton(
                            text: _isSignUp ? textLocalize('login_signup_btn') : textLocalize('login_login'),
                            type: TDButtonType.fill,
                            theme: TDButtonTheme.primary,
                            shape: TDButtonShape.round,
                            size: TDButtonSize.large,
                            isBlock: true,
                            onTap: _handleAuth,
                          ),
                    const SizedBox(height: 16),
                    TDButton(
                      text: _isSignUp ? textLocalize('login_goto_login') : textLocalize('login_goto_signup'),
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
