# scripts/create_db.py

from src.database.database import engine
from src.database.models import Base

def create_database():
    Base.metadata.create_all(bind=engine)
    print("Database created with all tables.")

if __name__ == "__main__":
    create_database()
