"""Explicit SQLAlchemy engine and session factory construction."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker


def create_engine_for_url(database_url: str) -> Engine:
    return create_engine(database_url, echo=False, future=True)


def create_session_factory(database_url: str) -> sessionmaker:
    engine = create_engine_for_url(database_url)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
