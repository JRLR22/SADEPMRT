from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

DATABASE_URL = settings.DATABASE_URL

# sqlite crea el archivo, pero no la carpeta que lo contiene
db_file = make_url(DATABASE_URL).database
if DATABASE_URL.startswith("sqlite") and db_file and db_file != ":memory:":
    Path(db_file).parent.mkdir(parents=True, exist_ok=True)

# fastapi usa la sesión desde varios hilos; check_same_thread solo existe en sqlite
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()