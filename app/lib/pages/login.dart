import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';

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
    return Scaffold(
      backgroundColor: TDTheme.of(context).grayColor1,
      appBar: AppBar(
        title: TDText(
          _isSignUp ? '注册' : '登录',
          font: TDTheme.of(context).fontHeadlineSmall,
          fontWeight: FontWeight.w600,
          textColor: TDTheme.of(context).fontGyColor1,
        ),
        backgroundColor: TDTheme.of(context).whiteColor1,
        elevation: 0,
        centerTitle: true,
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24.0),
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
            decoration: BoxDecoration(
              color: TDTheme.of(context).whiteColor1.withValues(alpha: 0.8),
              borderRadius: BorderRadius.circular(TDTheme.of(context).radiusExtraLarge),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.05),
                  blurRadius: 20,
                  spreadRadius: 5,
                )
              ],
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
    );
  }
}
