import time

from jose import jwt

from app.core import security
from config import settings


def test_hash_de_contrasena():
    h = security.get_password_hash("una-contrasena")
    assert h != "una-contrasena"
    assert security.verify_password("una-contrasena", h)
    assert not security.verify_password("otra", h)


def test_token_usa_la_clave_y_la_caducidad_de_settings():
    assert security.SECRET_KEY == settings.SECRET_KEY
    token = security.create_access_token({"sub": "u1"})
    assert security.decode_access_token(token)["sub"] == "u1"
    minutes = (jwt.get_unverified_claims(token)["exp"] - time.time()) / 60
    assert abs(minutes - settings.ACCESS_TOKEN_EXPIRE_MINUTES) < 1


def test_token_firmado_con_otra_clave_se_rechaza():
    token = jwt.encode({"sub": "u1", "exp": int(time.time()) + 60}, "otra-clave", algorithm="HS256")
    assert security.decode_access_token(token) is None


def test_token_caducado_se_rechaza():
    token = jwt.encode({"sub": "u1", "exp": int(time.time()) - 5}, settings.SECRET_KEY, algorithm="HS256")
    assert security.decode_access_token(token) is None
