import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/pages/community/models.dart';
import 'package:braindance/pages/settabs/settab1.dart';
import 'package:braindance/pages/settabs/settab3.dart';
import 'package:braindance/widgets/animated_network_image.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class MyOverviewTab extends StatelessWidget {
  final String userLabel;
  final CommunityStats stats;
  final bool isLoading;
  final VoidCallback onOpenCommunity;
  final VoidCallback onRefresh;

  const MyOverviewTab({
    super.key,
    required this.userLabel,
    required this.stats,
    required this.isLoading,
    required this.onOpenCommunity,
    required this.onRefresh,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.88);

    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          BDPanelCard(
            padding: const EdgeInsets.all(18),
            child: Row(
              children: [
                CircleAvatar(
                  radius: 28,
                  backgroundColor: BDDesign.colorMutedBlue.withValues(
                    alpha: isDark ? 0.20 : 0.12,
                  ),
                  child: Icon(
                    Icons.person_rounded,
                    color: BDDesign.colorMutedBlue,
                    size: 30,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        textLocalize('my_profile_title'),
                        style: TextStyle(
                          color: textColor,
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        userLabel,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(color: hintColor, fontSize: 13),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  onPressed: onRefresh,
                  icon: isLoading
                      ? SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: BDDesign.colorMutedBlue,
                          ),
                        )
                      : Icon(
                          Icons.refresh_rounded,
                          color: BDDesign.colorMutedBlue,
                        ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          GridView.count(
            crossAxisCount: 2,
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            childAspectRatio: 1.42,
            mainAxisSpacing: 10,
            crossAxisSpacing: 10,
            children: [
              _MetricTile(
                label: textLocalize('my_posts'),
                value: '${stats.postCount}',
                icon: Icons.article_outlined,
              ),
              _MetricTile(
                label: textLocalize('my_views'),
                value: '${stats.viewCount}',
                icon: Icons.visibility_outlined,
              ),
              _MetricTile(
                label: textLocalize('my_likes_received'),
                value: '${stats.likeCount}',
                icon: Icons.favorite_outline_rounded,
              ),
              _MetricTile(
                label: textLocalize('my_favorites_received'),
                value: '${stats.favoriteCount}',
                icon: Icons.star_outline_rounded,
              ),
              _MetricTile(
                label: textLocalize('my_comments'),
                value: '${stats.commentCount}',
                icon: Icons.chat_bubble_outline_rounded,
              ),
              _MetricTile(
                label: textLocalize('my_shareable_models'),
                value: '${stats.shareableModelCount}',
                icon: Icons.view_in_ar_rounded,
              ),
            ],
          ),
          const SizedBox(height: 12),
          BDPanelCard(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    stats.draftCount > 0
                        ? textLocalize('my_draft_ready')
                        : textLocalize('my_no_draft'),
                    style: TextStyle(
                      color: textColor,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                FilledButton.icon(
                  onPressed: onOpenCommunity,
                  icon: const Icon(Icons.groups_rounded, size: 18),
                  label: Text(textLocalize('my_open_community')),
                  style: FilledButton.styleFrom(
                    shape: RoundedRectangleBorder(
                      borderRadius: BDDesign.radiusLarge,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class MyCommunityTab extends StatelessWidget {
  final List<CommunityPost> myPosts;
  final List<CommunityPost> favoritePosts;
  final List<CommunityPost> likedPosts;
  final CommunityDraft draft;
  final bool isLoading;
  final ValueChanged<CommunityPost> onOpenPost;
  final ValueChanged<CommunityPost> onDeletePost;
  final ValueChanged<CommunityPost> onToggleVisibility;
  final VoidCallback onContinueDraft;
  final VoidCallback onRefresh;

  const MyCommunityTab({
    super.key,
    required this.myPosts,
    required this.favoritePosts,
    required this.likedPosts,
    required this.draft,
    required this.isLoading,
    required this.onOpenPost,
    required this.onDeletePost,
    required this.onToggleVisibility,
    required this.onContinueDraft,
    required this.onRefresh,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (!draft.isEmpty) ...[
            _DraftCard(draft: draft, onContinueDraft: onContinueDraft),
            const SizedBox(height: 12),
          ],
          _PostSection(
            title: textLocalize('my_post_management'),
            posts: myPosts,
            emptyText: textLocalize('my_empty_posts'),
            isLoading: isLoading,
            onRefresh: onRefresh,
            onOpenPost: onOpenPost,
            onDeletePost: onDeletePost,
            onToggleVisibility: onToggleVisibility,
            manageable: true,
          ),
          const SizedBox(height: 12),
          _PostSection(
            title: textLocalize('my_favorites'),
            posts: favoritePosts,
            emptyText: textLocalize('my_empty_favorites'),
            isLoading: isLoading,
            onRefresh: onRefresh,
            onOpenPost: onOpenPost,
          ),
          const SizedBox(height: 12),
          _PostSection(
            title: textLocalize('my_liked_posts'),
            posts: likedPosts,
            emptyText: textLocalize('my_empty_likes'),
            isLoading: isLoading,
            onRefresh: onRefresh,
            onOpenPost: onOpenPost,
          ),
        ],
      ),
    );
  }
}

class MySettingsTab extends StatelessWidget {
  final WidgetRef ref;

  const MySettingsTab({super.key, required this.ref});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        setTab1(context, ref),
        setTab3(context),
      ],
    );
  }
}

class _MetricTile extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;

  const _MetricTile({
    required this.label,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.58)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.78);

    return BDPanelCard(
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: BDDesign.colorMutedBlue, size: 20),
          const Spacer(),
          Text(
            value,
            style: TextStyle(
              color: textColor,
              fontSize: 24,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 3),
          Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              color: hintColor,
              fontSize: 12.5,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _PostSection extends StatelessWidget {
  final String title;
  final List<CommunityPost> posts;
  final String emptyText;
  final bool isLoading;
  final VoidCallback onRefresh;
  final ValueChanged<CommunityPost> onOpenPost;
  final ValueChanged<CommunityPost>? onDeletePost;
  final ValueChanged<CommunityPost>? onToggleVisibility;
  final bool manageable;

  const _PostSection({
    required this.title,
    required this.posts,
    required this.emptyText,
    required this.isLoading,
    required this.onRefresh,
    required this.onOpenPost,
    this.onDeletePost,
    this.onToggleVisibility,
    this.manageable = false,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.58)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.78);

    return BDPanelCard(
      padding: const EdgeInsets.fromLTRB(14, 14, 14, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  title,
                  style: TextStyle(
                    color: textColor,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              Text(
                '${posts.length}',
                style: TextStyle(
                  color: hintColor,
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          if (isLoading && posts.isEmpty)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 18),
              child: Center(child: CircularProgressIndicator()),
            )
          else if (posts.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 18),
              child: Center(
                child: Text(
                  emptyText,
                  textAlign: TextAlign.center,
                  style: TextStyle(color: hintColor, height: 1.4),
                ),
              ),
            )
          else
            ...posts.map(
              (post) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: _PostTile(
                  post: post,
                  onTap: () => onOpenPost(post),
                  manageable: manageable,
                  onDelete: onDeletePost == null
                      ? null
                      : () => onDeletePost!(post),
                  onToggleVisibility: onToggleVisibility == null
                      ? null
                      : () => onToggleVisibility!(post),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _PostTile extends StatelessWidget {
  final CommunityPost post;
  final VoidCallback onTap;
  final bool manageable;
  final VoidCallback? onDelete;
  final VoidCallback? onToggleVisibility;

  const _PostTile({
    required this.post,
    required this.onTap,
    required this.manageable,
    this.onDelete,
    this.onToggleVisibility,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.56)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.76);

    return Material(
      color: Colors.transparent,
      borderRadius: BDDesign.radiusLarge,
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(8),
          child: Row(
            children: [
              _PostThumbnail(imageUrl: post.coverUrl),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      post.title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: textColor,
                        fontSize: 14.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      '${post.placeName} · ${post.relativeTimeLabel}',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(color: hintColor, fontSize: 12),
                    ),
                    const SizedBox(height: 7),
                    Wrap(
                      spacing: 8,
                      runSpacing: 6,
                      children: [
                        _InlineMetric(
                          icon: Icons.visibility_outlined,
                          value: post.viewCount,
                        ),
                        _InlineMetric(
                          icon: Icons.favorite_outline_rounded,
                          value: post.likeCount,
                        ),
                        _InlineMetric(
                          icon: Icons.star_outline_rounded,
                          value: post.favoriteCount,
                        ),
                        _InlineMetric(
                          icon: Icons.chat_bubble_outline_rounded,
                          value: post.commentCount,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              if (manageable)
                PopupMenuButton<String>(
                  icon: Icon(Icons.more_horiz_rounded, color: hintColor),
                  onSelected: (value) {
                    if (value == 'visibility') {
                      onToggleVisibility?.call();
                    }
                    if (value == 'delete') {
                      onDelete?.call();
                    }
                  },
                  itemBuilder: (context) => [
                    PopupMenuItem(
                      value: 'visibility',
                      child: Text(
                        post.isPublic
                            ? textLocalize('my_make_private')
                            : textLocalize('my_make_public'),
                      ),
                    ),
                    PopupMenuItem(
                      value: 'delete',
                      child: Text(textLocalize('my_delete_post')),
                    ),
                  ],
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _InlineMetric extends StatelessWidget {
  final IconData icon;
  final int value;

  const _InlineMetric({required this.icon, required this.value});

  @override
  Widget build(BuildContext context) {
    final color = context.isDarkMode
        ? Colors.white.withValues(alpha: 0.58)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.82);
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: color),
        const SizedBox(width: 3),
        Text(
          '$value',
          style: TextStyle(
            color: color,
            fontSize: 11.5,
            fontWeight: FontWeight.w700,
          ),
        ),
      ],
    );
  }
}

class _PostThumbnail extends StatelessWidget {
  final String? imageUrl;

  const _PostThumbnail({required this.imageUrl});

  @override
  Widget build(BuildContext context) {
    final fallback = Container(
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFF8BA8C5),
            Color(0xFF536C8B),
            Color(0xFF38485F),
          ],
        ),
        borderRadius: BorderRadius.circular(18),
      ),
      child: const Icon(Icons.terrain_rounded, color: Colors.white, size: 24),
    );
    final url = imageUrl;
    return SizedBox(
      width: 70,
      height: 66,
      child: url == null || url.isEmpty
          ? fallback
          : BDFadeInNetworkImage(
              imageUrl: url,
              placeholder: fallback,
              errorWidget: fallback,
              borderRadius: BorderRadius.circular(18),
              fit: BoxFit.cover,
              duration: BDMotion.durationSlow,
            ),
    );
  }
}

class _DraftCard extends StatelessWidget {
  final CommunityDraft draft;
  final VoidCallback onContinueDraft;

  const _DraftCard({required this.draft, required this.onContinueDraft});

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.58)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.78);

    return BDPanelCard(
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          Icon(Icons.edit_note_rounded, color: BDDesign.colorMutedBlue),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  textLocalize('my_draft_box'),
                  style: TextStyle(
                    color: textColor,
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  draft.title.isEmpty
                      ? textLocalize('my_untitled_draft')
                      : draft.title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(color: hintColor, fontSize: 12.5),
                ),
              ],
            ),
          ),
          TextButton(
            onPressed: onContinueDraft,
            child: Text(textLocalize('my_continue_draft')),
          ),
        ],
      ),
    );
  }
}
