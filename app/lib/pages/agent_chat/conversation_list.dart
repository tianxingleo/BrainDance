part of '../agent_chat.dart';

extension _AgentChatConversationList on _AgentChatPageState {
  Widget _buildConversationDrawer(bool isDark) {
    final bgColor = isDark ? const Color(0xFF141A24) : Colors.white;
    final textColor = isDark ? Colors.white : BDDesign.colorInkBlack;
    final hintColor = isDark
        ? Colors.white.withValues(alpha: 0.5)
        : Colors.black.withValues(alpha: 0.45);
    final selectedColor = isDark
        ? Colors.white.withValues(alpha: 0.08)
        : BDDesign.colorMutedBlue.withValues(alpha: 0.08);

    return Drawer(
      backgroundColor: bgColor,
      child: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      textLocalize('agent_conversations'),
                      style: TextStyle(
                        color: textColor,
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  IconButton(
                    icon: Icon(
                      Icons.add_rounded,
                      color: textColor,
                      size: 22,
                    ),
                    onPressed: () {
                      Navigator.pop(context);
                      _createNewConversation();
                    },
                  ),
                ],
              ),
            ),
            const Divider(height: 1),
            Expanded(
              child: _conversationList.isEmpty
                  ? Center(
                      child: Text(
                        textLocalize('agent_no_conversations'),
                        style: TextStyle(color: hintColor, fontSize: 14),
                      ),
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      itemCount: _conversationList.length,
                      itemBuilder: (context, index) {
                        final conv = _conversationList[index];
                        final isSelected =
                            conv.id == _currentConversation?.id;
                        return _buildConversationTile(
                          conv,
                          isSelected: isSelected,
                          isDark: isDark,
                          textColor: textColor,
                          hintColor: hintColor,
                          selectedColor: selectedColor,
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildConversationTile(
    AgentConversation conv, {
    required bool isSelected,
    required bool isDark,
    required Color textColor,
    required Color hintColor,
    required Color selectedColor,
  }) {
    final title = conv.title.isNotEmpty
        ? conv.title
        : textLocalize('agent_untitled_conversation');
    final timeAgo = _formatTimeAgo(conv.updatedAt);

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: isSelected ? selectedColor : Colors.transparent,
        borderRadius: BorderRadius.circular(10),
      ),
      child: ListTile(
        dense: true,
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
        title: Text(
          title,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: TextStyle(
            color: textColor,
            fontSize: 14,
            fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
          ),
        ),
        subtitle: Text(
          timeAgo,
          style: TextStyle(color: hintColor, fontSize: 12),
        ),
        trailing: PopupMenuButton<String>(
          icon: Icon(Icons.more_horiz, color: hintColor, size: 18),
          itemBuilder: (context) => [
            PopupMenuItem(
              value: 'delete',
              child: Row(
                children: [
                  const Icon(Icons.delete_outline, size: 18, color: Colors.red),
                  const SizedBox(width: 8),
                  Text(textLocalize('agent_delete_conversation')),
                ],
              ),
            ),
          ],
          onSelected: (value) {
            if (value == 'delete') {
              _deleteConversation(conv.id);
            }
          },
        ),
        onTap: () {
          Navigator.pop(context);
          if (!isSelected) {
            _loadConversation(conv);
          }
        },
      ),
    );
  }

  String _formatTimeAgo(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inMinutes < 1) return '刚刚';
    if (diff.inMinutes < 60) return '${diff.inMinutes}分钟前';
    if (diff.inHours < 24) return '${diff.inHours}小时前';
    if (diff.inDays < 7) return '${diff.inDays}天前';
    return '${dt.month}/${dt.day}';
  }
}
