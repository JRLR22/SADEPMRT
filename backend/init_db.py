"""Crea las tablas y un usuario administrador inicial. Se ejecuta una sola vez."""

import secrets
import sys
from pathlib import Path

# para poder importar app/ aunque se ejecute desde otra carpeta
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from app.database.base import Base, engine
from app.models.user import User
from app.models.study import Study, Prediction, AuditLog
from app.core.security import get_password_hash
from sqlalchemy.orm import Session
from loguru import logger

# el valor de .env.example es público: nunca debe quedar como contraseña real
PLACEHOLDER_ADMIN_PASSWORD = "CAMBIA-ESTA-CLAVE-DE-ADMIN"


def init_database():
    """Crea todas las tablas en la base de datos"""

    logger.info("Iniciando creación de base de datos...")

    try:
        logger.info("Creando tablas...")
        Base.metadata.create_all(bind=engine)
        logger.success("Tablas creadas exitosamente")

        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        logger.info(f"Tablas en la base de datos: {', '.join(tables)}")

        logger.info("Creando usuario administrador...")
        create_admin_user()

        logger.success("Base de datos inicializada correctamente")

    except Exception as e:
        logger.error(f"Error inicializando base de datos: {e}")
        raise


def create_admin_user():
    """Crea el administrador inicial si todavía no existe."""

    from sqlalchemy.orm import Session

    with Session(engine) as session:
        from app.models.user import User, UserRole
        existing_admin = session.query(User).filter(
            User.email == settings.ADMIN_EMAIL
        ).first()

        if existing_admin:
            logger.info(f"El usuario {settings.ADMIN_EMAIL} ya existe; no se cambió su contraseña.")
            return

        password = settings.ADMIN_PASSWORD
        generated = password in ("", PLACEHOLDER_ADMIN_PASSWORD)
        if generated:
            password = secrets.token_urlsafe(16)

        admin = User(
            email=settings.ADMIN_EMAIL,
            hashed_password=get_password_hash(password),
            full_name="Administrador del Sistema",
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True
        )

        session.add(admin)
        session.commit()

        logger.success(f"Usuario administrador creado: {settings.ADMIN_EMAIL}")

        # con print y no con logger, para que la contraseña no acabe en un archivo de logs
        if generated:
            print(f"\nContraseña generada (se muestra solo esta vez): {password}\n")
        else:
            logger.info("Contraseña tomada de ADMIN_PASSWORD")


if __name__ == "__main__":
    logger.info("Iniciando script de inicialización...")
    # misma conexión que la api: DATABASE_URL, o sqlite en ./lung_detection.db por defecto
    logger.info(f"Conectando a: {engine.url}")

    init_database()