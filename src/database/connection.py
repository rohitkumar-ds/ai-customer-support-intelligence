from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine,event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker,Session
from src.config import get_settings

@event.listens_for(Engine,"connect")
def configure_sqlite(dbapi_connection, connection_record):
    """Enable SQLite integrity/performance pragmas on new connections."""
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cur=dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        try: cur.execute("PRAGMA journal_mode=WAL")
        except Exception: pass
        cur.execute("PRAGMA synchronous=NORMAL"); cur.close()

_engine=None
_Session=None

def get_engine():
    """Create the global SQLAlchemy engine lazily."""
    global _engine,_Session
    if _engine is None:
        s=get_settings() 
        url=s.database.url
        if url.startswith("sqlite:///") and ":memory:" not in url:
            raw=url.replace("sqlite:///","") 
            Path(raw).parent.mkdir(parents=True,exist_ok=True)
        _engine=create_engine(url,echo=s.database.echo_sql,connect_args={"check_same_thread":False} if url.startswith("sqlite") else {})
        _Session=sessionmaker(bind=_engine,autoflush=False,autocommit=False)
    return _engine

@contextmanager
def get_db()->Generator[Session,None,None]:
    """Yield a transaction-scoped database session."""
    session=get_session()
    try:
        yield session; session.commit()
    except Exception:
        session.rollback()
        raise
    finally: session.close()

def get_session()->Session:
    """Return a new SQLAlchemy session."""
    get_engine() 
    return _Session()

def init_db(engine:Engine|None=None):
    """Create all ORM tables if they do not exist."""
    from src.database.models import Base
    Base.metadata.create_all(engine or get_engine())
