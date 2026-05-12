import 'package:braindance/configs/app_config.dart';
import 'package:braindance/configs/app_theme.dart';
import 'package:braindance/configs/motion_tokens.dart';
import 'package:braindance/services/viewer_navigation.dart';
import 'package:braindance/widgets/bd_surfaces.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'models.dart';
import 'repository.dart';

class CommunityDetailPage extends StatefulWidget {
  final CommunityPost post;

  const CommunityDetailPage({super.key, required this.post});

  @override
  State<CommunityDetailPage> createState() => _CommunityDetailPageState();
}

class _CommunityDetailPageState extends State<CommunityDetailPage> {
  final CommunityRepository _repository = CommunityRepository();
  final ScrollController _scrollController = ScrollController();
  final TextEditingController _commentController = TextEditingController();

  late CommunityPost _post;
  List<CommunityComment> _comments = const [];
  Map<String, dynamic> _metadata = {};
  bool _isLiked = false;
  bool _isFavorited = false;

  @override
  void initState() {
    super.initState();
    _post = widget.post;
    _loadMetadata();
    _loadComments();
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _commentController.dispose();
    super.dispose();
  }

  Future<void> _loadMetadata() async {
    final meta = await _repository.fetchPostMetadata(_post.id);
    if (!mounted) return;
    final uid = _repository.currentUserId;
    setState(() {
      _metadata = meta;
      _isLiked = (meta['likes'] as List?)?.contains(uid) ?? false;
      _isFavorited =
          (meta['favorites'] as List?)?.contains(uid) ?? false;
      // Sync counts from server metadata so they never go stale
      _post = _post.copyWith(
        likeCount: (meta['likes'] as List?)?.length ?? 0,
        favoriteCount: (meta['favorites'] as List?)?.length ?? 0,
        commentCount: (meta['comments'] as List?)?.length ?? 0,
      );
    });
  }

  Future<void> _loadComments() async {
    final comments = await _repository.fetchComments(_post.id);
    if (!mounted) return;
    setState(() => _comments = comments);
  }

  void _toggleLike() {
    final uid = _repository.currentUserId;
    final likes = List<String>.from(_metadata['likes'] ?? []);
    final wasLiked = likes.contains(uid);
    if (wasLiked) {
      likes.remove(uid);
    } else {
      likes.add(uid);
    }

    // Optimistic UI update — fire immediately, no waiting
    final optimisticMeta = {..._metadata, 'likes': likes};
    setState(() {
      _metadata = optimisticMeta;
      _isLiked = !wasLiked;
      _post = _post.copyWith(likeCount: likes.length);
    });

    // Async persist in background
    _repository.setMetadata(_post.id, optimisticMeta);
  }

  void _toggleFavorite() {
    final uid = _repository.currentUserId;
    final favorites =
        List<String>.from(_metadata['favorites'] ?? []);
    final wasFavorited = favorites.contains(uid);
    if (wasFavorited) {
      favorites.remove(uid);
    } else {
      favorites.add(uid);
    }

    // Optimistic UI update
    final optimisticMeta = {
      ..._metadata,
      'favorites': favorites,
    };
    setState(() {
      _metadata = optimisticMeta;
      _isFavorited = !wasFavorited;
      _post = _post.copyWith(favoriteCount: favorites.length);
    });

    // Async persist in background
    _repository.setMetadata(_post.id, optimisticMeta);
  }

  void _submitComment() {
    final text = _commentController.text.trim();
    if (text.isEmpty) return;
    _commentController.clear();

    final now = DateTime.now();
    final uid = _repository.currentUserId;
    final userName =
        Supabase.instance.client.auth.currentUser?.email ?? '匿名用户';
    final optimisticComment = CommunityComment(
      id: 'c-$uid-${now.microsecondsSinceEpoch}',
      postId: _post.id,
      userId: uid,
      userName: userName,
      text: text,
      createdAt: now,
    );

    // Optimistic UI update — show comment immediately
    final comments = <CommunityComment>[optimisticComment, ..._comments];
    setState(() {
      _comments = comments;
      _post = _post.copyWith(commentCount: comments.length);
    });

    // Async persist in background, then sync metadata
    _repository.addComment(_post.id, text, _metadata).then((_) {
      if (!mounted) return;
      _repository.fetchPostMetadata(_post.id).then((meta) {
        if (!mounted) return;
        setState(() => _metadata = meta);
      });
    });
  }

  void _scrollToComments() {
    if (_scrollController.hasClients) {
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: BDMotion.durationNormal,
        curve: BDMotion.curveFluid,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.55)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.72);

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          Column(
            children: [
              Expanded(
                child: BDPageBackdrop(
              child: SafeArea(
                bottom: false,
                child: SingleChildScrollView(
                  controller: _scrollController,
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox(
                      height: 320,
                      child: _DetailThumbnail(
                        imageUrl: _post.coverUrl,
                        height: 320,
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 12,
                      ),
                      child: SizedBox(
                        width: double.infinity,
                        child: FilledButton.icon(
                          onPressed: _enterViewer,
                          icon: const Icon(
                            Icons.view_in_ar_rounded,
                          ),
                          label: Text(
                            textLocalize('community_enter_memory'),
                          ),
                          style: FilledButton.styleFrom(
                            padding: const EdgeInsets.symmetric(
                              vertical: 14,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BDDesign.radiusLarge,
                            ),
                          ),
                        ),
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                      ),
                      child: Column(
                        crossAxisAlignment:
                            CrossAxisAlignment.start,
                        children: [
                          Text(
                            _post.title,
                            style: TextStyle(
                              color: textColor,
                              fontSize: 22,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            '${_post.authorName} · ${_post.placeName} · ${_post.relativeTimeLabel}',
                            style: TextStyle(
                              color: hintColor,
                              fontSize: 13,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                      ),
                      child: BDPanelCard(
                        padding: const EdgeInsets.all(16),
                        child: Column(
                          crossAxisAlignment:
                              CrossAxisAlignment.start,
                          children: [
                            Text(
                              '作品简介',
                              style: TextStyle(
                                color: textColor,
                                fontSize: 15,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              _post.caption,
                              style: TextStyle(
                                color: textColor
                                    .withValues(alpha: 0.82),
                                height: 1.5,
                              ),
                            ),
                            const SizedBox(height: 12),
                            Wrap(
                              spacing: 6,
                              runSpacing: 6,
                              children: _post.tags.map((tag) {
                                return Container(
                                  padding:
                                      const EdgeInsets.symmetric(
                                    horizontal: 10,
                                    vertical: 4,
                                  ),
                                  decoration: BoxDecoration(
                                    color: isDark
                                        ? Colors.white
                                            .withValues(
                                              alpha: 0.06,
                                            )
                                        : BDDesign.colorMutedBlue
                                            .withValues(
                                              alpha: 0.08,
                                            ),
                                    borderRadius:
                                        BorderRadius.circular(12),
                                  ),
                                  child: Text(
                                    tag,
                                    style: TextStyle(
                                      color:
                                          BDDesign.colorMutedBlue,
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                );
                              }).toList(),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                      ),
                      child: Text(
                        '评论 (${_post.commentCount})',
                        style: TextStyle(
                          color: textColor,
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                    if (_comments.isEmpty)
                      Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20,
                        ),
                        child: Text(
                          '暂无评论，来发布第一条评论吧',
                          style: TextStyle(color: hintColor),
                        ),
                      )
                    else
                      ..._comments.map((comment) {
                        return Padding(
                          padding: const EdgeInsets.fromLTRB(
                            20,
                            0,
                            20,
                            10,
                          ),
                          child: _CommentTile(
                            comment: comment,
                            isDark: isDark,
                          ),
                        );
                      }),
                  ],
                ),
              ),
            ),
            ),
          ),
          Container(
            padding: EdgeInsets.fromLTRB(
              16,
              10,
              16,
              10 + MediaQuery.of(context).padding.bottom,
            ),
            decoration: BoxDecoration(
              color: isDark
                  ? AppTheme.darkSurface.withValues(alpha: 0.96)
                  : Colors.white.withValues(alpha: 0.94),
              border: Border(
                top: BorderSide(
                      color: isDark
                          ? Colors.white.withValues(alpha: 0.06)
                          : BDDesign.colorMutedBlue
                              .withValues(alpha: 0.08),
                    ),
                  ),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _commentController,
                        decoration: InputDecoration(
                          hintText: '发表评论...',
                          filled: true,
                          fillColor: isDark
                              ? AppTheme.darkSurfaceElevated
                              : const Color(0xFFF3F5F9),
                          border: OutlineInputBorder(
                            borderRadius: BDDesign.radiusLarge,
                            borderSide: BorderSide.none,
                          ),
                          contentPadding:
                              const EdgeInsets.symmetric(
                            horizontal: 14,
                            vertical: 10,
                          ),
                        ),
                        onSubmitted: (_) => _submitComment(),
                      ),
                    ),
                    const SizedBox(width: 8),
                    _ActionButton(
                      icon: _isLiked
                          ? Icons.favorite_rounded
                          : Icons.favorite_outline_rounded,
                      label: '${_post.likeCount}',
                      color: _isLiked ? Colors.red : null,
                      onTap: _toggleLike,
                    ),
                    _ActionButton(
                      icon: _isFavorited
                          ? Icons.star_rounded
                          : Icons.star_outline_rounded,
                      label: '${_post.favoriteCount}',
                      color: _isFavorited
                          ? const Color(0xFFEAC86B)
                          : null,
                      onTap: _toggleFavorite,
                    ),
                    _ActionButton(
                      icon: Icons.chat_bubble_outline_rounded,
                      label: '${_post.commentCount}',
                      onTap: _scrollToComments,
                    ),
                  ],
                ),
              ),
            ],
          ),
          _buildBackButton(),
        ],
      ),
    );
  }

  Widget _buildBackButton() {
    return Positioned(
      left: 16,
      top: MediaQuery.of(context).padding.top + 8,
      child: GestureDetector(
        onTap: () => Navigator.pop(context),
        child: Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            color: Colors.black.withValues(alpha: 0.35),
            shape: BoxShape.circle,
          ),
          child: const Icon(
            Icons.arrow_back_rounded,
            color: Colors.white,
            size: 22,
          ),
        ),
      ),
    );
  }

  void _enterViewer() {
    openViewer(
      context,
      initialModelUrl: _post.modelUrl,
      posesUrl: _post.posesUrl,
      sceneId: _post.modelName,
    );
  }
}

class _DetailThumbnail extends StatelessWidget {
  final String? imageUrl;
  final double height;

  const _DetailThumbnail({
    required this.imageUrl,
    required this.height,
  });

  @override
  Widget build(BuildContext context) {
    final fallback = Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFF8BA8C5),
            Color(0xFF536C8B),
            Color(0xFF38485F),
          ],
        ),
      ),
      child: Center(
        child: Icon(
          Icons.terrain_rounded,
          size: 48,
          color: Colors.white.withValues(alpha: 0.7),
        ),
      ),
    );

    final url = imageUrl;
    if (url == null || url.isEmpty) {
      return SizedBox(height: height, child: fallback);
    }
    return SizedBox(
      height: height,
      width: double.infinity,
      child: Image.network(
        url,
        fit: BoxFit.cover,
        errorBuilder: (_, _, _) => fallback,
      ),
    );
  }
}

class _CommentTile extends StatelessWidget {
  final CommunityComment comment;
  final bool isDark;

  const _CommentTile({
    required this.comment,
    required this.isDark,
  });

  @override
  Widget build(BuildContext context) {
    final textColor =
        isDark ? BDDesign.colorPaperWhite : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.55)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.72);
    final initial =
        comment.userName.isNotEmpty ? comment.userName[0] : '?';

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        CircleAvatar(
          radius: 16,
          backgroundColor: isDark
              ? BDDesign.colorMutedBlue.withValues(alpha: 0.18)
              : BDDesign.colorMutedBlue.withValues(alpha: 0.10),
          child: Text(
            initial,
            style: TextStyle(
              color: BDDesign.colorMutedBlue,
              fontSize: 13,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text(
                    comment.userName,
                    style: TextStyle(
                      color: textColor,
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    comment.relativeTimeLabel,
                    style: TextStyle(color: hintColor, fontSize: 11),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                comment.text,
                style: TextStyle(
                  color: textColor.withValues(alpha: 0.82),
                  height: 1.35,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color? color;
  final VoidCallback onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = context.isDarkMode;
    final defaultColor = isDark
        ? Colors.white.withValues(alpha: 0.62)
        : BDDesign.colorMutedBlue;

    return GestureDetector(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 6),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: color ?? defaultColor, size: 22),
            const SizedBox(height: 2),
            Text(
              label,
              style: TextStyle(
                color: color ?? defaultColor,
                fontSize: 10,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
