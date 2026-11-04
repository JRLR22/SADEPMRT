"""Registro público y gestión de roles por un administrador."""

import json

import pytest

from app.database.base import SessionLocal
from app.models.study import AuditLog

NEW_USER = {"password": "contrasena-larga", "full_name": "Usuario Nuevo"}


def register(client, email, **extra):
    return client.post("/auth/register", json={"email": email, **NEW_USER, **extra})


def user_id(client, headers):
    return client.get("/auth/me", headers=headers).json()["id"]


@pytest.mark.parametrize("role", ["admin", "radiologist", "viewer"])
def test_el_registro_publico_no_permite_elegir_rol(client, role):
    r = register(client, f"intento-{role}@pruebas.com", role=role)
    assert r.status_code == 422
    assert client.post("/auth/login", json={"email": f"intento-{role}@pruebas.com",
                                            "password": NEW_USER["password"]}).status_code == 401


def test_el_registro_publico_crea_viewer(client):
    r = register(client, "lector@pruebas.com")
    assert r.status_code == 201 and r.json()["role"] == "viewer"


def test_solo_un_admin_lista_usuarios(client, make_user):
    admin = make_user("admin")
    r = client.get("/auth/users", headers=admin)
    assert r.status_code == 200 and user_id(client, admin) in [u["id"] for u in r.json()]
    for role in ("viewer", "radiologist"):
        assert client.get("/auth/users", headers=make_user(role)).status_code == 403


def test_admin_asigna_rol_y_el_cambio_aplica_al_momento(client, make_user, xray_bytes):
    admin, user = make_user("admin"), make_user()
    uid = user_id(client, user)

    r = client.put(f"/auth/users/{uid}/role", json={"role": "radiologist"}, headers=admin)
    assert r.status_code == 200 and r.json()["role"] == "radiologist"
    # el rol se lee de la base en cada petición: el mismo token ya tiene los permisos nuevos
    assert client.get("/auth/me", headers=user).json()["role"] == "radiologist"

    sid = client.post("/predict/auto", headers=user, files={"file": ("rx.jpg", xray_bytes, "image/jpeg")}).json()["study_id"]
    pred_id = client.get(f"/studies/{sid}", headers=user).json()["predictions"][0]["id"]
    assert client.put(f"/predictions/{pred_id}/validate", params={"validation_result": "ok"},
                      headers=user).status_code == 200

    with SessionLocal() as db:
        audit = db.query(AuditLog).filter(AuditLog.action == "role_change").order_by(AuditLog.created_at.desc()).first()
    assert json.loads(audit.meta_data) == {"user_id": uid, "from": "viewer", "to": "radiologist"}
    assert audit.user_id == user_id(client, admin)


def test_quien_no_es_admin_no_cambia_roles(client, make_user):
    target = user_id(client, make_user())
    for role in ("viewer", "radiologist"):
        headers = make_user(role)
        assert client.put(f"/auth/users/{target}/role", json={"role": "admin"}, headers=headers).status_code == 403
        # tampoco puede ascenderse a sí mismo
        assert client.put(f"/auth/users/{user_id(client, headers)}/role", json={"role": "admin"},
                          headers=headers).status_code == 403


def test_admin_no_cambia_su_propio_rol(client, make_user):
    admin = make_user("admin")
    r = client.put(f"/auth/users/{user_id(client, admin)}/role", json={"role": "viewer"}, headers=admin)
    assert r.status_code == 400
    assert client.get("/auth/me", headers=admin).json()["role"] == "admin"


def test_usuario_inexistente_o_rol_invalido(client, make_user):
    admin = make_user("admin")
    assert client.put("/auth/users/no-existe/role", json={"role": "viewer"}, headers=admin).status_code == 404
    target = user_id(client, make_user())
    assert client.put(f"/auth/users/{target}/role", json={"role": "superusuario"}, headers=admin).status_code == 422
