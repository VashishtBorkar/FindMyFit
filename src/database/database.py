# src/database/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from dotenv import load_dotenv
import os

from src.database.models import Base

load_dotenv()
database = os.getenv("DATABASE_URL", "sqlite:///data/findmyfit.db")
print("DATABASE_URL from env:", repr(os.getenv("DATABASE_URL")))
print("Final database value:", repr(database))
print("Using database at:", database)

engine = create_engine(database, echo=False, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
