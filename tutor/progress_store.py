"""
Task 6: local progress in SQLite. Learner rows keyed by HMAC(learner_id).

**At-rest encryption (optional):** set ``TUTOR_USE_ENCRYPTED_DB=1`` and (recommended)
``TUTOR_DB_KEY`` to a long random passphrase. The on-disk file is
``<name>.db.crypt`` (Fernet); a decrypted copy lives in a temp path only while the
process is open, and is re-encrypted after writes / on close. If unset, plain
``<name>.db`` is used (dev default).

Set ``TUTOR_HMAC_SECRET`` in production for stable learner keys.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import hmac
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from cryptography.fernet import Fernet

_KEY_ENV = "TUTOR_HMAC_SECRET"
_DEFAULT = b"tutor-day3-dev-only-change-in-prod"
_DB_SALT = b"day3-progress-db-fernet-v1"


def _use_encrypted_storage() -> bool:
    if (os.environ.get("TUTOR_DB_KEY", "") or "").strip():
        return True
    v = (os.environ.get("TUTOR_USE_ENCRYPTED_DB", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _fernet() -> "Fernet | None":
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        return None
    raw = (os.environ.get("TUTOR_DB_KEY", "") or "").strip()
    if not raw:
        sec = (os.environ.get(_KEY_ENV, "") or "").encode("utf-8")
        if not sec or sec == _DEFAULT:
            return None
        raw = hmac.new(_DB_SALT, sec, hashlib.sha256).hexdigest()[:32]
    key = base64.urlsafe_b64encode(hashlib.sha256(raw.encode("utf-8")).digest())
    return Fernet(key)


@dataclass
class Attempt:
    ts: float
    item_id: str
    subskill: str
    correct: int


def _learner_key(learner_id: str) -> bytes:
    sec = os.environ.get(_KEY_ENV, "").encode("utf-8") or _DEFAULT
    return hmac.new(sec, learner_id.encode("utf-8"), hashlib.sha256).digest()


def _anon_tag(learner_id: str) -> str:
    h = hashlib.sha256(f"day3|{learner_id}".encode()).hexdigest()[:10]
    return f"learner_{h}"


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            learner_key BLOB NOT NULL,
            item_id TEXT NOT NULL,
            subskill TEXT NOT NULL,
            correct INTEGER NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_ts ON attempts (ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lk ON attempts (learner_key)")


def _enc_paths(base: Path) -> tuple[Path, Path]:
    """Encrypted blob at ``progress.db.crypt`` when *base* is ``.../progress.db``."""
    crypted = Path(str(base) + ".crypt")
    return crypted, base


class ProgressDB:
    def __init__(self, path: str | Path | None = None) -> None:
        self._base = Path(
            path or (Path(__file__).resolve().parents[1] / "tutor" / "outputs" / "progress.db")
        )
        self._base.parent.mkdir(parents=True, exist_ok=True)
        _want = _use_encrypted_storage()
        self._f = _fernet() if _want else None
        if _want and self._f is None:
            raise RuntimeError(
                "Encrypted progress DB requested (TUTOR_USE_ENCRYPTED_DB=1 and/or TUTOR_DB_KEY) but "
                "Fernet could not be built: install `cryptography` and set TUTOR_DB_KEY to a long "
                "passphrase, or TUTOR_HMAC_SECRET to a non-default value, or TUTOR_USE_ENCRYPTED_DB=0."
            )

        self._crypted_path, self._legacy_plain = _enc_paths(self._base)
        self._work: Path
        self._enc_mode = self._f is not None
        self._closed = False
        if self._enc_mode:
            fd, name = tempfile.mkstemp(
                prefix="tutor_pdb_",
                suffix=".db",
                dir=str(self._base.parent),
            )
            os.close(fd)
            self._work = Path(name)
            if self._crypted_path.is_file():
                self._work.write_bytes(self._f.decrypt(self._crypted_path.read_bytes()))
            elif self._legacy_plain.is_file() and not self._crypted_path.is_file():
                # migrate: encrypt existing plain dev DB
                self._work.write_bytes(self._legacy_plain.read_bytes())
            self._conn = sqlite3.connect(str(self._work))
            atexit.register(self._sync_encrypt_atexit)
        else:
            self._work = self._base
            self._conn = sqlite3.connect(str(self._work))
        _init_schema(self._conn)
        self._conn.commit()

    def _sync_encrypt(self) -> None:
        if not self._enc_mode or self._f is None:
            return
        self._conn.commit()
        data = self._work.read_bytes()
        tmp = Path(str(self._crypted_path) + ".tmp")
        tmp.write_bytes(self._f.encrypt(data))
        tmp.replace(self._crypted_path)
        if self._legacy_plain.is_file():
            try:
                self._legacy_plain.unlink()
            except OSError:
                pass

    def _sync_encrypt_atexit(self) -> None:
        if getattr(self, "_closed", False):
            return
        try:
            self._sync_encrypt()
        except OSError:
            pass
        try:
            if self._enc_mode and self._work.is_file():
                self._work.unlink(missing_ok=True)
        except OSError:
            pass

    def _after_write(self) -> None:
        self._conn.commit()
        if self._enc_mode:
            self._sync_encrypt()

    def log_attempt(
        self, learner_id: str, item_id: str, subskill: str, correct: bool
    ) -> None:
        b = _learner_key(learner_id)
        self._conn.execute(
            "INSERT INTO attempts (ts, learner_key, item_id, subskill, correct) VALUES (?,?,?,?,?)",
            (time.time(), b, item_id, subskill, 1 if correct else 0),
        )
        self._after_write()

    def iter_attempts_for_learner(
        self, learner_id: str, t0: float, t1: float
    ) -> Iterator[Attempt]:
        b = _learner_key(learner_id)
        cur = self._conn.execute(
            "SELECT ts, item_id, subskill, correct FROM attempts WHERE learner_key = ? AND ts >= ? AND ts <= ?",
            (b, t0, t1),
        )
        for row in cur:
            yield Attempt(float(row[0]), str(row[1]), str(row[2]), int(row[3]))

    def close(self) -> None:
        if self._enc_mode:
            self._sync_encrypt()
        self._conn.close()
        self._closed = True
        if self._enc_mode and self._work.is_file():
            self._work.unlink(missing_ok=True)

    def on_disk_path_for_docs(self) -> str:
        """Path shown in logging (encrypted blob or plain file)."""
        if self._enc_mode:
            return str(self._crypted_path)
        return str(self._work)


_db_singleton: ProgressDB | None = None


def get_progress_db(path: str | Path | None = None) -> ProgressDB:
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = ProgressDB(path)
    return _db_singleton
