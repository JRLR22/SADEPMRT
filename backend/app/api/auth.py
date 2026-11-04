"""Endpoints de autenticación: registro, login y usuario actual."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from loguru import logger

from app.database.base import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, UserResponse, Token
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


router = APIRouter(prefix="/auth", tags=["Autenticación"])
security = HTTPBearer()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Registra un usuario nuevo con el rol indicado."""
    logger.info(f"Intento de registro: {user_data.email}")

    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        logger.warning(f"Email ya registrado: {user_data.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Este email ya está registrado"
        )

    if user_data.license_number:
        existing_license = db.query(User).filter(
            User.license_number == user_data.license_number
        ).first()
        if existing_license:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Este número de licencia ya está registrado"
            )

    new_user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        role=user_data.role,
        license_number=user_data.license_number,
        institution=user_data.institution,
        specialty=user_data.specialty,
        is_active=True,
        is_verified=False  # aún no hay verificación por correo
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.success(f"Usuario registrado: {new_user.email} ({new_user.role})")

    return new_user


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Inicia sesión con email y contraseña y devuelve un token JWT."""
    logger.info(f"Intento de login: {credentials.email}")

    user = db.query(User).filter(User.email == credentials.email).first()

    if not user:
        logger.warning(f"Usuario no encontrado: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos"
        )

    if not verify_password(credentials.password, user.hashed_password):
        logger.warning(f"Contraseña incorrecta para: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos"
        )

    if not user.is_active:
        logger.warning(f"Usuario inactivo: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Usuario desactivado. Contacta al administrador."
        )

    access_token = create_access_token(data={"sub": str(user.id)})

    from datetime import datetime
    user.last_login = datetime.now()
    db.commit()

    logger.success(f"Login exitoso: {user.email}")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


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
    """Devuelve el usuario dueño del token enviado en Authorization."""
    logger.info(f"Usuario consultando su perfil: {current_user.email}")
    return current_user


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Cierra sesión; el token se borra en el cliente, aquí solo queda registro."""
    logger.info(f"Usuario cerrando sesión: {current_user.email}")

    return {"message": "Sesión cerrada exitosamente"}