import sqlite3, time, json, os
DB_PATH = os.environ.get("EVENTS_DB", "telemetry.db")

def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS events(
          id INTEGER PRIMARY KEY,
          user_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          type TEXT NOT NULL,         -- 'search' | 'click'
          query TEXT,
          product_id TEXT,
          meta TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
        """)

def add_event(user_id: str, etype: str, query: str | None = None, product_id: str | None = None, meta: dict | None = None, ts: int | None = None):
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            "INSERT INTO events(user_id, ts, type, query, product_id, meta) VALUES(?,?,?,?,?,?)",
            (user_id, ts or int(time.time()), etype, query, product_id, json.dumps(meta or {}))
        )

def get_recent_events(user_id: str, since_secs: int = 60*60*24*90, limit: int = 4000):
    with sqlite3.connect(DB_PATH) as c:
        cutoff = int(time.time()) - since_secs
        cur = c.execute(
            "SELECT ts,type,query,product_id,meta FROM events WHERE user_id=? AND ts>=? ORDER BY ts DESC LIMIT ?",
            (user_id, cutoff, limit)
        )
        return cur.fetchall()
