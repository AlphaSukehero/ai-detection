"""Patient registry and study store.

Plain functions over a sqlite3 connection plus a file store on disk. Callers
get dicts, never cursors. Studies are append-only.
"""
