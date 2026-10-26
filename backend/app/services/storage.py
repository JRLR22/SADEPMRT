# ============================================================================
# STORAGE.PY - ALMACENAMIENTO LOCAL (SIN SUPABASE)
# ============================================================================

import os
from pathlib import Path
from loguru import logger
from typing import Optional


class StorageService:
    """Servicio para gestión de imágenes en disco local"""

    def __init__(self):
        self.base_dir = Path("uploads")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "xrays").mkdir(exist_ok=True)
        (self.base_dir / "gradcam").mkdir(exist_ok=True)
        (self.base_dir / "segmentation").mkdir(exist_ok=True)
        (self.base_dir / "yolo").mkdir(exist_ok=True)
        logger.info("✅ StorageService local inicializado")

    def upload_image(self, image_bytes: bytes, filename: str, folder: str = "xrays") -> str:
        """
        Guarda imagen en disco local.
        Retorna URL local accesible por el frontend.
        """
        try:
            full_path = self.base_dir / folder / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_path, "wb") as f:
                f.write(image_bytes)

            local_url = f"http://localhost:8000/uploads/{folder}/{filename}"
            logger.success(f"✅ Imagen guardada: {full_path}")
            return local_url

        except Exception as e:
            logger.error(f"❌ Error guardando imagen: {e}")
            raise Exception(f"No se pudo guardar la imagen: {e}")

    def delete_image(self, file_path: str) -> bool:
        """Elimina imagen del disco"""
        try:
            path = Path("uploads") / file_path.replace("http://localhost:8000/uploads/", "")
            if path.exists():
                path.unlink()
                logger.success(f"🗑️ Imagen eliminada: {path}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error eliminando imagen: {e}")
            return False

    def get_image_url(self, file_path: str) -> Optional[str]:
        """Retorna URL local si el archivo existe"""
        try:
            path = self.base_dir / file_path
            if path.exists():
                return f"http://localhost:8000/uploads/{file_path}"
            return None
        except Exception as e:
            logger.error(f"❌ Error obteniendo URL: {e}")
            return None