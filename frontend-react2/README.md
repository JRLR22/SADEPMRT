# Frontend · rt26

Interfaz React (Create React App) del sistema de diagnóstico pulmonar. La descripción del proyecto, los
datasets y las limitaciones están en el [README principal](../README.md).

```powershell
npm install
npm start        # http://localhost:3000
npm test         # prueba de humo del login
npm run build    # compilación de producción en build/
```

Variables de entorno (opcionales, en `frontend-react2/.env`):

| Variable | Para qué sirve |
| --- | --- |
| `REACT_APP_API_URL` | URL del backend. Por defecto `http://localhost:8000` |
| `REACT_APP_ENABLE_CHAT` | `true` activa el widget de chat (requiere un servicio externo en el puerto 8001; desactivado por defecto) |
| `REACT_APP_DEMO_EMAIL` / `REACT_APP_DEMO_PASSWORD` | Muestran el botón "Probar demo sin cuenta". Usa una cuenta de solo lectura: estos valores quedan visibles en el JavaScript publicado |

Estructura de `src/`:

- `App.js`: pantallas (login, análisis, dashboard, historial, pacientes, panel de radiólogo).
- `ChatWidget.js`: asistente de chat flotante, opcional y desactivado por defecto.
- `ui.css`: interacciones, animaciones y diseño responsive.
- `public/samples/`: radiografías de ejemplo de datasets públicos.
