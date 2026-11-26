"""MySQL storage for user feedback (kept for analysis and future model retraining).

Works with MySQL 8 (MySQL Installer, XAMPP, WAMP, Laragon) and MariaDB 10.4+.
The database and the `feedback` table are created automatically on first start.
"""
from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import pymysql

COLUMNS = ("id", "created_at", "source_text", "source_lang", "target_lang",
           "generated_translation", "corrected_translation", "explanation", "model_version")

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id                    INT UNSIGNED  NOT NULL AUTO_INCREMENT PRIMARY KEY,
    created_at            DATETIME      NOT NULL,
    source_text           TEXT          NOT NULL,
    source_lang           VARCHAR(16)   NOT NULL,
    target_lang           VARCHAR(16)   NOT NULL,
    generated_translation TEXT          NOT NULL,
    corrected_translation TEXT          NOT NULL,
    explanation           TEXT          NULL,
    model_version         VARCHAR(255)  NULL,
    INDEX idx_feedback_direction (source_lang, target_lang),
    CONSTRAINT chk_feedback_langs CHECK (source_lang <> target_lang)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]{1,64}$")


class DatabaseUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class MySQLConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "bikole"

    def __post_init__(self):
        if not _IDENTIFIER.match(self.database):
            raise ValueError("MYSQL_DATABASE may only contain letters, numbers and underscores.")


class FeedbackRepository:
    def __init__(self, config: MySQLConfig):
        self.config = config
        self.ready = False
        self.error: str | None = None
        self._init_lock = threading.Lock()

    def _connect(self, with_database: bool = True):
        c = self.config
        return pymysql.connect(
            host=c.host, port=c.port, user=c.user, password=c.password,
            database=c.database if with_database else None,
            charset="utf8mb4", connect_timeout=5, autocommit=False,
        )

    @contextmanager
    def _connection(self):
        if not self.ready:
            self.init()
        try:
            conn = self._connect()
        except pymysql.MySQLError as exc:
            raise DatabaseUnavailable(self._describe(exc)) from exc
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _describe(self, exc: Exception) -> str:
        c = self.config
        code = exc.args[0] if exc.args else None
        if code == 2003:
            return f"Cannot connect to MySQL at {c.host}:{c.port}. Is the MySQL server running?"
        if code == 1045:
            return f"MySQL refused user '{c.user}'. Check MYSQL_USER / MYSQL_PASSWORD in the .env file."
        return f"MySQL error: {exc}"

    def init(self) -> None:
        """Create the database and table if needed. Safe to call repeatedly."""
        with self._init_lock:
            if self.ready:
                return
            try:
                conn = self._connect(with_database=False)
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.config.database}` "
                                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                        cur.execute(f"USE `{self.config.database}`")
                        cur.execute(TABLE_SQL)
                    conn.commit()
                finally:
                    conn.close()
            except pymysql.MySQLError as exc:
                self.ready, self.error = False, self._describe(exc)
                raise DatabaseUnavailable(self.error) from exc
            self.ready, self.error = True, None

    def try_init(self) -> bool:
        try:
            self.init()
            return True
        except DatabaseUnavailable:
            return False

    def add(self, *, source_text: str, source_lang: str, target_lang: str, generated_translation: str,
            corrected_translation: str, explanation: str | None, model_version: str | None) -> int:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (created_at, source_text, source_lang, target_lang, generated_translation, "
                "corrected_translation, explanation, model_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0), source_text, source_lang, target_lang,
                 generated_translation, corrected_translation, explanation, model_version),
            )
            return int(cur.lastrowid)

    def list_all(self) -> list[dict]:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {', '.join(COLUMNS)} FROM feedback ORDER BY id")
            rows = cur.fetchall()
        return [dict(zip(COLUMNS, r)) for r in rows]

    def count(self) -> int:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM feedback")
            return int(cur.fetchone()[0])

    def info(self) -> dict:
        c = self.config
        return {"engine": "mysql", "host": c.host, "port": c.port, "database": c.database,
                "state": "ready" if self.ready else "unavailable", "error": self.error}
