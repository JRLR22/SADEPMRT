import pytest

import config
from config import Settings, validate_config


def test_allowed_origins_acepta_comas_json_o_valor_por_defecto(monkeypatch):
    assert Settings(ALLOWED_ORIGINS="https://a.com, http://b.com").ALLOWED_ORIGINS == ["https://a.com", "http://b.com"]
    monkeypatch.setenv("ALLOWED_ORIGINS", '["https://a.com","http://b.com"]')
    assert Settings().ALLOWED_ORIGINS == ["https://a.com", "http://b.com"]
    monkeypatch.delenv("ALLOWED_ORIGINS")
    assert "http://localhost:3000" in Settings().ALLOWED_ORIGINS


def test_variables_desconocidas_del_env_se_ignoran(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("FRONTEND_PORT=80\nDOMAIN=tu-dominio.com\nACCESS_TOKEN_EXPIRE_MINUTES=15\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ACCESS_TOKEN_EXPIRE_MINUTES")
    assert Settings().ACCESS_TOKEN_EXPIRE_MINUTES == 15


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    """Apunta todas las rutas de modelos a archivos temporales que existen."""
    for name in [n for n in dir(config.settings) if n.endswith("_PATH")]:
        path = tmp_path / f"{name.lower()}.pth"
        path.write_bytes(b"x")
        monkeypatch.setattr(config.settings, name, path)
    return tmp_path


def test_clave_de_ejemplo_solo_avisa_en_desarrollo(model_files, monkeypatch, capsys):
    monkeypatch.setattr(config.settings, "SECRET_KEY", "cambia-esta-clave-en-produccion")
    validate_config()
    assert "AVISO: SECRET_KEY no definido" in capsys.readouterr().out


@pytest.mark.parametrize("key", ["", "change-me-in-production", "CAMBIA-ESTO-POR-UNA-CLAVE-SEGURA"])
def test_clave_de_ejemplo_o_vacia_impide_arrancar_en_produccion(model_files, monkeypatch, key):
    monkeypatch.setattr(config.settings, "APP_ENV", "production")
    monkeypatch.setattr(config.settings, "SECRET_KEY", key)
    with pytest.raises(RuntimeError, match="SECRET_KEY no definido"):
        validate_config()


def test_produccion_con_clave_propia_arranca(model_files, monkeypatch, capsys):
    monkeypatch.setattr(config.settings, "APP_ENV", "production")
    validate_config()
    assert "Configuración validada" in capsys.readouterr().out


def test_falta_un_modelo_obligatorio(model_files, monkeypatch):
    monkeypatch.setattr(config.settings, "UNET_MODEL_PATH", model_files / "no-existe.pth")
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        validate_config()


def test_yolo_es_opcional(model_files, monkeypatch, capsys):
    monkeypatch.setattr(config.settings, "YOLO_MODEL_PATH", model_files / "no-existe.pt")
    validate_config()
    out = capsys.readouterr().out
    assert "detección YOLO queda desactivada" in out and "Configuración validada" in out
