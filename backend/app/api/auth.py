"""
ENDPOINTS DE AUTENTICACIÓN
===========================

Login, Register, Get Current User
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta
from loguru import logger

from app.database.base import get_db
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserLogin, UserResponse, Token
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid


router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


# ============================================================================
# REGISTER - Crear nuevo usuario
# ============================================================================

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Registra un nuevo usuario en el sistema.
    
    **Ejemplo:**
    ```json
    {
      "email": "doctor@hospital.com",
      "password": "SecurePass123!",
      "full_name": "Dr. Juan Pérez",
      "role": "radiologist",
      "license_number": "MED-12345",
      "institution": "Hospital General"
    }
    ```
    """
    
    logger.info(f"Intento de registro: {user_data.email}")
    
    # Verificar si el email ya existe
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        logger.warning(f" Email ya registrado: {user_data.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Este email ya está registrado"
        )
    
    # Verificar si el número de licencia ya existe (si se proporciona)
    if user_data.license_number:
        existing_license = db.query(User).filter(
            User.license_number == user_data.license_number
        ).first()
        if existing_license:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Este número de licencia ya está registrado"
            )
    
    # Crear usuario
    new_user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        role=user_data.role,
        license_number=user_data.license_number,
        institution=user_data.institution,
        specialty=user_data.specialty,
        is_active=True,
        is_verified=False  # Requerirá verificación de email en producción
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    logger.success(f"Usuario registrado: {new_user.email} ({new_user.role})")
    
    return new_user


# ============================================================================
# LOGIN - Autenticar usuario
# ============================================================================

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Inicia sesión y devuelve un token JWT.
    
    **Ejemplo:**
    ```json
    {
      "email": "admin@hospital.com",
      "password": "admin123"
    }
    ```
    
    **Respuesta:**
    ```json
    {
      "access_token": "eyJhbGciOiJIUzI1NiIs...",
      "token_type": "bearer",
      "user": { ... }
    }
    ```
    """
    
    logger.info(f"Intento de login: {credentials.email}")
    
    # Buscar usuario por email
    user = db.query(User).filter(User.email == credentials.email).first()
    
    if not user:
        logger.warning(f"Usuario no encontrado: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos"
        )
    
    # Verificar contraseña
    if not verify_password(credentials.password, user.hashed_password):
        logger.warning(f"Contraseña incorrecta para: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos"
        )
    
    # Verificar que el usuario esté activo
    if not user.is_active:
        logger.warning(f"Usuario inactivo: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuario desactivado. Contacta al administrador."
        )
    
    # Crear token JWT
    access_token = create_access_token(data={"sub": str(user.id)})
    
    # Actualizar last_login
    from datetime import datetime
    user.last_login = datetime.now()
    db.commit()
    
    logger.success(f"Login exitoso: {user.email}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


# ============================================================================
# GET CURRENT USER - Obtener usuario actual desde token
# ============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token inválido")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Usuario desactivado")

    return user


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Obtiene la información del usuario actual.
    
    **Requiere:** Token JWT en header Authorization: Bearer <token>
    """
    
    logger.info(f"Usuario consultando su perfil: {current_user.email}")
    return current_user


# ============================================================================
# LOGOUT (opcional - el logout real es client-side borrando el token)
# ============================================================================

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout (opcional).
    
    En JWT, el logout real es client-side (borrar el token del localStorage).
    Este endpoint solo sirve para logging/auditoría.
    """
    
    logger.info(f"Usuario cerrando sesión: {current_user.email}")
    
    return {"message": "Sesión cerrada exitosamente"}