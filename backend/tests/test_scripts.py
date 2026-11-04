"""init_db.py y download_models.py, ejecutados contra carpetas temporales."""

import hashlib
import os
import re
import sqlite3
import subprocess
import sys

import pytest

import download_models
from conftest import BACKEND

GENERATED = re.compile(r"Contraseña generada \(se muestra solo esta vez\): (\S+)")


def run_init_db(tmp_path, **env):
    full_env = {k: v for k, v in os.environ.items() if not k.startswith("ADMIN_")}
    full_env.update(DATABASE_URL=f"sqlite:///{(tmp_path / 'init.db').as_posix()}", PYTHONUTF8="1", **env)
    r = subprocess.run([sys.executable, str(BACKEND / "init_db.py")], cwd=tmp_path, env=full_env,
                       capture_output=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = sqlite3.connect(tmp_path / "init.db").execute("select email, hashed_password from users").fetchall()
    return r.stdout + r.stderr, rows


def test_init_db_genera_contrasena_y_la_muestra_una_vez(tmp_path):
    from app.core.security import verify_password
    out, rows = run_init_db(tmp_path)
    shown = GENERATED.findall(out)
    assert len(shown) == 1 and rows[0][0] == "admin@hospital.com"
    assert verify_password(shown[0], rows[0][1])

    out, rows_again = run_init_db(tmp_path)
    assert not GENERATED.search(out) and "ya existe" in out
    assert rows_again == rows


def test_init_db_usa_admin_password_sin_mostrarla(tmp_path):
    from app.core.security import verify_password
    out, rows = run_init_db(tmp_path, ADMIN_EMAIL="jefe@clinica.com", ADMIN_PASSWORD="clave-propia-9f3k")
    assert "clave-propia-9f3k" not in out and not GENERATED.search(out)
    assert rows[0][0] == "jefe@clinica.com" and verify_password("clave-propia-9f3k", rows[0][1])


def test_init_db_no_acepta_la_contrasena_de_ejemplo(tmp_path):
    from app.core.security import verify_password
    out, rows = run_init_db(tmp_path, ADMIN_PASSWORD="CAMBIA-ESTA-CLAVE-DE-ADMIN")
    assert GENERATED.search(out) and not verify_password("CAMBIA-ESTA-CLAVE-DE-ADMIN", rows[0][1])


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    hub, models = tmp_path / "hub", tmp_path / "models"
    hub.mkdir(), models.mkdir()
    files = {}
    for name in ("a.pth", "b.pth"):
        (hub / name).write_bytes(name.encode() * 100)
        files[name] = hashlib.sha256((hub / name).read_bytes()).hexdigest()
    calls = []

    def fake_download(repo_id, filename, revision, local_dir):
        calls.append((repo_id, filename, revision))
        if not (hub / filename).exists():
            raise RuntimeError("404")
        (models / filename).write_bytes((hub / filename).read_bytes())

    monkeypatch.setattr(download_models, "hf_hub_download", fake_download)
    monkeypatch.setattr(download_models, "MODELS_DIR", models)
    monkeypatch.setattr(download_models, "MODELS", files)
    return hub, models, calls


def run_download(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["download_models.py", *args])
    return download_models.main()


def test_descarga_solo_lo_que_falta(fake_hub, monkeypatch):
    hub, models, calls = fake_hub
    assert run_download(monkeypatch) == 0 and len(calls) == 2
    assert calls[0][0] == download_models.REPO_ID and calls[0][2] == download_models.REVISION
    calls.clear()
    (models / "a.pth").unlink()
    assert run_download(monkeypatch) == 0 and [c[1] for c in calls] == ["a.pth"]


def test_descarga_con_hash_incorrecto_se_borra(fake_hub, monkeypatch):
    hub, models, _ = fake_hub
    (hub / "b.pth").write_bytes(b"alterado")
    assert run_download(monkeypatch) == 1 and not (models / "b.pth").exists()


def test_verify_no_borra_archivos_propios(fake_hub, monkeypatch):
    hub, models, calls = fake_hub
    (models / "a.pth").write_bytes(b"mi modelo")
    (models / "b.pth").write_bytes((hub / "b.pth").read_bytes())
    assert run_download(monkeypatch) == 0 and not calls
    assert run_download(monkeypatch, "--verify") == 1 and (models / "a.pth").read_bytes() == b"mi modelo"


def test_fallo_de_descarga(fake_hub, monkeypatch):
    hub, _, _ = fake_hub
    (hub / "a.pth").unlink()
    assert run_download(monkeypatch) == 1


def test_el_script_fija_un_commit_y_todos_los_modelos_obligatorios():
    from config import settings
    assert re.fullmatch(r"[0-9a-f]{40}", download_models.REVISION)
    required = {p.name for n, p in vars(settings).items() if n.endswith("_PATH") and "YOLO" not in n}
    assert set(download_models.MODELS) == required
