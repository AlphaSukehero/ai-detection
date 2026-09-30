"""SQLite connection and numbered schema migrations."""
import os
import sqlite3

DB_NAME = "records.db"

# Append only. Each entry runs once, in order, and bumps schema_version.
MIGRATIONS = [
    """
    CREATE TABLE patients (
        seq         INTEGER PRIMARY KEY AUTOINCREMENT,
        id          TEXT NOT NULL UNIQUE,
        name        TEXT NOT NULL,
        dob         TEXT,
        sex         TEXT,
        phone       TEXT,
        address     TEXT,
        notes       TEXT,
        created_at  TEXT NOT NULL
    );
    CREATE TABLE studies (
        id                   TEXT PRIMARY KEY,
        patient_id           TEXT NOT NULL REFERENCES patients(id),
        modality             TEXT NOT NULL,
        study_date           TEXT NOT NULL,
        created_at           TEXT NOT NULL,
        referring_physician  TEXT,
        clinical_history     TEXT,
        verdict              TEXT,
        headline             TEXT,
        model_name           TEXT,
        model_version        TEXT,
        result_json          TEXT NOT NULL,
        report_json          TEXT NOT NULL,
        reanalysis_of        TEXT REFERENCES studies(id)
    );
    CREATE INDEX studies_by_patient ON studies(patient_id, modality, study_date);
    CREATE TABLE measurements (
        study_id  TEXT NOT NULL REFERENCES studies(id),
        key       TEXT NOT NULL,
        value     REAL,
        unit      TEXT,
        PRIMARY KEY (study_id, key)
    );
    CREATE TABLE study_files (
        study_id  TEXT NOT NULL REFERENCES studies(id),
        kind      TEXT NOT NULL,
        path      TEXT NOT NULL,
        PRIMARY KEY (study_id, kind)
    );
    """,
]


def connect(root):
    """Open (creating if needed) the store under `root` and migrate it."""
    os.makedirs(root, exist_ok=True)
    conn = sqlite3.connect(os.path.join(root, DB_NAME), isolation_level=None,
                           check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


def migrate(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    version = row[0] if row else 0
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
    for number, sql in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in filter(str.strip, sql.split(";")):
                conn.execute(statement)
            conn.execute("UPDATE schema_version SET version = ?", (number,))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


class transaction:
    """`with transaction(conn):` -- BEGIN IMMEDIATE ... COMMIT / ROLLBACK.

    IMMEDIATE takes the write lock up front, so ID allocation (read max, then
    insert) cannot race another writer.
    """

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        return False
