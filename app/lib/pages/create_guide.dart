import 'package:flutter/material.dart';
import '../configs/app_config.dart';
import '../configs/app_theme.dart';
import '../configs/motion_tokens.dart';
import '../widgets/bd_surfaces.dart';
import 'record.dart';
import 'generate.dart';
import 'community.dart';

PageRoute<T> _verticalSlideRoute<T>({
  required WidgetBuilder builder,
  required bool fromTop, // true = 新页面从上进（相机），false = 从下进（图文生成）
}) {
  return PageRouteBuilder<T>(
    transitionDuration: BDMotion.durationNormal,
    reverseTransitionDuration: BDMotion.durationNormal,
    opaque: true,
    pageBuilder: (context, animation, secondaryAnimation) => builder(context),
    transitionsBuilder: (context, animation, secondaryAnimation, child) {
      final double dir = fromTop ? -1.0 : 1.0;
      // 新页面入场：从 dir 方向滑入
      final enterAnim = CurvedAnimation(
        parent: animation,
        curve: Curves.easeInOutCubic,
      );
      return SlideTransition(
        position: Tween<Offset>(
          begin: Offset(0, dir),
          end: Offset.zero,
        ).animate(enterAnim),
        child: child,
      );
    },
  );
}

/// Create 引导页 — record 和 generate 的统一入口
class CreateGuidePage extends StatelessWidget {
  const CreateGuidePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      extendBody: true,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.only(bottom: 96.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              BDPageHeader(
                title: textLocalize('create'),
                subtitle: textLocalize('create_guide_subtitle'),
              ),
              const SizedBox(height: 12),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Column(
                  children: [
                    _buildEntryCard(
                      context,
                      icon: Icons.camera_rounded,
                      title: textLocalize('record'),
                      subtitle: textLocalize('create_record_desc'),
                      onTap: () => Navigator.push(
                        context,
                        _verticalSlideRoute(
                          builder: (_) => const RecordPage(),
                          fromTop: true,
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    _buildEntryCard(
                      context,
                      icon: Icons.auto_awesome_rounded,
                      title: textLocalize('generate'),
                      subtitle: textLocalize('create_generate_desc'),
                      onTap: () => Navigator.push(
                        context,
                        _verticalSlideRoute(
                          builder: (_) => const GeneratePage(),
                          fromTop: false,
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    _buildEntryCard(
                      context,
                      icon: Icons.groups_rounded,
                      title: textLocalize('community'),
                      subtitle: textLocalize('create_community_desc'),
                      onTap: () => Navigator.push(
                        context,
                        _verticalSlideRoute(
                          builder: (_) => const CommunityPage(),
                          fromTop: false,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildEntryCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    final isDark = context.isDarkMode;
    final iconBg = isDark
        ? BDDesign.colorMutedBlue.withValues(alpha: 0.18)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.10);
    final iconColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorMutedBlue;
    final textColor = isDark
        ? BDDesign.colorPaperWhite
        : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.55)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.72);

    return GestureDetector(
      onTap: onTap,
      child: BDPanelCard(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            Container(
              width: 52,
              height: 52,
              decoration: BoxDecoration(
                color: iconBg,
                borderRadius: BDDesign.radiusSmall,
              ),
              child: Icon(icon, color: iconColor, size: 26),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      color: textColor,
                      fontSize: 17,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: TextStyle(color: hintColor, fontSize: 13),
                  ),
                ],
              ),
            ),
            Icon(Icons.chevron_right_rounded, color: hintColor, size: 22),
          ],
        ),
      ),
    );
  }
}
