"""Límite de peticiones por usuario y operación."""

import collections
import threading
import time

from fastapi import Depends, HTTPException, Request

from app.api.auth import get_current_user
from app.models.user import User


class RateLimiter:
    """Limitador de peticiones en memoria con ventana deslizante."""
    def __init__(self):
        self._store: dict[str, collections.deque] = {}
        self._lock  = threading.Lock()

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
        """Devuelve (permitido, segundos hasta poder reintentar)."""
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            if key not in self._store:
                self._store[key] = collections.deque()

            dq = self._store[key]

            while dq and dq[0] < cutoff:
                dq.popleft()

            if len(dq) >= max_requests:
                oldest = dq[0]
                retry_after = int(window_seconds - (now - oldest)) + 1
                return False, retry_after

            dq.append(now)
            return True, 0

    def cleanup(self):
        """Borra las claves sin actividad en la última hora."""
        now = time.time()
        with self._lock:
            dead = [k for k, dq in self._store.items() if not dq or dq[-1] < now - 3600]
            for k in dead:
                del self._store[k]


_rate_limiter = RateLimiter()

RATE_LIMITS = {
    "predict":    {"max_requests": 30,  "window_seconds": 3600},
    "report":     {"max_requests": 20,  "window_seconds": 3600},
    "studies":    {"max_requests": 200, "window_seconds": 3600},
    "validate":   {"max_requests": 100, "window_seconds": 3600},
}


def check_rate_limit(request: Request, current_user: User, operation: str = "predict"):
    """Lanza 429 si el usuario superó el límite de esa operación."""
    cfg = RATE_LIMITS.get(operation, RATE_LIMITS["predict"])
    # por cuenta y no por ip
    key = f"{operation}:{current_user.id}"

    allowed, retry_after = _rate_limiter.is_allowed(
        key, cfg["max_requests"], cfg["window_seconds"]
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error":        "rate_limit_exceeded",
                "message":      f"Límite de {cfg['max_requests']} requests/{cfg['window_seconds']//3600}h superado.",
                "retry_after":  retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )


# depends no permite pasar la operación como argumento: una función por cada una
def rate_limit_predict(request: Request, current_user: User = Depends(get_current_user)):
    check_rate_limit(request, current_user, "predict")

def rate_limit_report(request: Request, current_user: User = Depends(get_current_user)):
    check_rate_limit(request, current_user, "report")

def rate_limit_studies(request: Request, current_user: User = Depends(get_current_user)):
    check_rate_limit(request, current_user, "studies")

def rate_limit_validate(request: Request, current_user: User = Depends(get_current_user)):
    check_rate_limit(request, current_user, "validate")
