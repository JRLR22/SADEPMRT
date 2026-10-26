"""
SCRIPT DE INICIALIZACIÓN DE BASE DE DATOS
==========================================

Este script crea todas las tablas necesarias en Supabase.
Solo necesitas ejecutarlo UNA VEZ.

CÓMO USAR:
1. Asegúrate de que tu .env tenga DATABASE_URL correcto
2. Ejecuta: python init_db.py
3. Verás "Base de datos inicializada"
"""

import sys
from pathlib import Path

# Agregar el directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

from app.database.base import Base, engine
from app.models.user import User
from app.models.study import Study, Prediction, AuditLog
from app.core.security import get_password_hash
from sqlalchemy.orm import Session
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

def init_database():
    """Crea todas las tablas en la base de datos"""
    
    logger.info("🗄️  Iniciando creación de base de datos...")
    
    try:
        # Crear todas las tablas
        logger.info("Creando tablas...")
        Base.metadata.create_all(bind=engine)
        logger.success("Tablas creadas exitosamente")
        
        # Verificar tablas creadas
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        logger.info(f"Tablas en la base de datos: {', '.join(tables)}")
        
        # Crear usuario administrador por defecto
        logger.info("Creando usuario administrador...")
        create_admin_user()
        
        logger.success("Base de datos inicializada correctamente")
        logger.info("""
╔══════════════════════════════════════════════════════════╗
║     BASE DE DATOS LISTA                                  ║
╠══════════════════════════════════════════════════════════╣
║  Usuario admin creado:                                   ║
║     Email:    admin@hospital.com                         ║
║     Password: admin123                                   ║
║                                                          ║
║      CAMBIA ESTA CONTRASEÑA EN PRODUCCIÓN                ║
╚══════════════════════════════════════════════════════════╝
        """)
        
    except Exception as e:
        logger.error(f"❌ Error inicializando base de datos: {e}")
        raise


def create_admin_user():
    """Crea un usuario administrador por defecto"""
    
    from sqlalchemy.orm import Session
    
    with Session(engine) as session:
        # Verificar si ya existe un admin
        from app.models.user import User, UserRole
        existing_admin = session.query(User).filter(
            User.email == "admin@hospital.com"
        ).first()
        
        if existing_admin:
            logger.info("ℹ️  Usuario admin ya existe, omitiendo...")
            return
        
        # Crear usuario admin
        admin = User(
            email="admin@hospital.com",
            hashed_password=get_password_hash("admin123"),
            full_name="Administrador del Sistema",
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True
        )
        
        session.add(admin)
        session.commit()
        
        logger.success("Usuario administrador creado")


if __name__ == "__main__":
    logger.info("Iniciando script de inicialización...")
    
    # Verificar que DATABASE_URL esté configurado
    db_url = os.getenv("DATABASE_URL")
    if not db_url or "sqlite" in db_url:
        logger.error("""
❌ ERROR: DATABASE_URL no configurado correctamente

Por favor edita tu archivo .env y agrega:
DATABASE_URL="postgresql://postgres.xxxxx:PASSWORD@....supabase.com:6543/postgres"

Obtén esta URL desde:
1. https://supabase.com/dashboard
2. Settings → Database → Connection string → URI
        """)
        sys.exit(1)
    
    logger.info(f" Conectando a: {db_url[:50]}...")
    
    init_database()