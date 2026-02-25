import sqlite3
from src.utils.config import get_config

class MemoryDB:
    def __init__(self):
        config = get_config()
        self.db_path = config["DB_PATH"]
        self.init_database()

    def init_database(self):
        db = sqlite3.connect(self.db_path, check_same_thread=False)
        db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_memory (
            session_id TEXT,
            role TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        db.commit()
        db.close()

    def store_memory(self, session_id, role, message):
        db = sqlite3.connect(self.db_path, check_same_thread=False)
        db.execute(
            "INSERT INTO conversation_memory (session_id, role, message) VALUES (?, ?, ?)",
            (session_id, role, message)
        )
        db.commit()
        db.close()

    def get_conversation_memory(self, session_id, limit=6):
        db = sqlite3.connect(self.db_path, check_same_thread=False)
        rows = db.execute(
            """
            SELECT role, message FROM conversation_memory
            WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?
            """,
            (session_id, limit)
        ).fetchall()
        db.close()

        return "\n".join(
            f"{role}: {msg}" for role, msg in reversed(rows)
        )


    def get_chat_history_tuples(self, session_id, limit = 6):
        db = sqlite3.connect(self.db_path, check_same_thread=False)
        rows = db.execute(
            """
            SELECT role, message
            FROM conversation_memory
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """,
            (session_id, )
        ).fetchall()
        db.close()

        history = []
        user_msg = None

        for role, msg in rows:
            if role == "user":
                user_msg = msg
            elif role == "assistant" and user_msg:
                history.append((user_msg, msg))
                user_msg = None

        return history[-limit:]

    def get_chat_history_text(self, session_id, limit = 6):
        db = sqlite3.connect(self.db_path, check_same_thread=False)
        rows = db.execute(
            """
            SELECT role, message
            FROM conversation_memory
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit)
        ).fetchall()
        db.close()

        return "\n".join(
            f"{role}: {msg}" for role, msg in reversed(rows)
        )
    def clear_memory(self, session_id):
        db = sqlite3.connect(self.db_path, check_same_thread=False)
        db.execute(
            "DELETE FROM conversation_memory WHERE session_id = ?",
            (session_id, )
        )
        db.commit()
        db.close()