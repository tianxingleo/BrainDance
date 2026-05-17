import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

import '../models/agent_conversation_model.dart';

class AgentConversationDb {
  AgentConversationDb._();
  static final AgentConversationDb instance = AgentConversationDb._();

  Database? _db;

  Future<Database> _database() async {
    if (_db != null) return _db!;
    final dir = await getApplicationDocumentsDirectory();
    final dbPath = path.join(dir.path, 'braindance_agent_conversations.db');
    _db = await openDatabase(
      dbPath,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            session_id TEXT,
            conversation_summary TEXT,
            session_state_json TEXT,
            short_term_memory_json TEXT
          )
        ''');
        await db.execute('''
          CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            is_user INTEGER NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            final_answer TEXT,
            timestamp INTEGER NOT NULL,
            agent_result_json TEXT,
            elapsed_ms INTEGER,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
              ON DELETE CASCADE
          )
        ''');
        await db.execute('''
          CREATE INDEX idx_messages_conversation
          ON messages(conversation_id, timestamp ASC)
        ''');
      },
    );
    return _db!;
  }

  // ── Conversations ──

  Future<AgentConversation> createConversation({
    required String id,
    String title = '',
  }) async {
    final db = await _database();
    final now = DateTime.now();
    final conv = AgentConversation(
      id: id,
      title: title,
      createdAt: now,
      updatedAt: now,
    );
    await db.insert('conversations', conv.toMap());
    return conv;
  }

  Future<List<AgentConversation>> listConversations({int limit = 50}) async {
    final db = await _database();
    final rows = await db.query(
      'conversations',
      orderBy: 'updated_at DESC',
      limit: limit,
    );
    return rows.map(AgentConversation.fromMap).toList();
  }

  Future<AgentConversation?> getConversation(String id) async {
    final db = await _database();
    final rows = await db.query(
      'conversations',
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );
    if (rows.isEmpty) return null;
    return AgentConversation.fromMap(rows.first);
  }

  Future<void> updateConversation(AgentConversation conv) async {
    final db = await _database();
    conv.updatedAt = DateTime.now();
    await db.update(
      'conversations',
      conv.toMap(),
      where: 'id = ?',
      whereArgs: [conv.id],
    );
  }

  Future<void> deleteConversation(String id) async {
    final db = await _database();
    await db.delete('messages', where: 'conversation_id = ?', whereArgs: [id]);
    await db.delete('conversations', where: 'id = ?', whereArgs: [id]);
  }

  // ── Messages ──

  Future<int> insertMessage(AgentMessageRecord msg) async {
    final db = await _database();
    final id = await db.insert('messages', msg.toMap());
    await db.update(
      'conversations',
      {'updated_at': msg.timestamp.millisecondsSinceEpoch},
      where: 'id = ?',
      whereArgs: [msg.conversationId],
    );
    return id;
  }

  Future<List<AgentMessageRecord>> getMessages(String conversationId) async {
    final db = await _database();
    final rows = await db.query(
      'messages',
      where: 'conversation_id = ?',
      whereArgs: [conversationId],
      orderBy: 'timestamp ASC',
    );
    return rows.map(AgentMessageRecord.fromMap).toList();
  }

  Future<void> updateMessage(int id, Map<String, dynamic> fields) async {
    final db = await _database();
    await db.update('messages', fields, where: 'id = ?', whereArgs: [id]);
  }
}
