from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional
from datetime import datetime
from app.models.user import UserRole


class UserBase(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=3, max_length=255)
    license_number: Optional[str] = None
    institution: Optional[str] = None
    specialty: Optional[str] = None


class UserCreate(UserBase):
    # el registro es público: el rol no se elige aquí, lo asigna un administrador
    model_config = ConfigDict(extra="forbid")

    password: str = Field(..., min_length=8, max_length=100)


class RoleUpdate(BaseModel):
    role: UserRole


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    license_number: Optional[str] = None
    institution: Optional[str] = None
    specialty: Optional[str] = None
    is_active: Optional[bool] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(UserBase):
    id: str        # uuid guardado como texto
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TokenPayload(BaseModel):
    sub: str
    exp: Optional[datetime] = None