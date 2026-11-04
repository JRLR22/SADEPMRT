# rt26 · Diagnóstico pulmonar asistido por IA

https://github.com/user-attachments/assets/a1798fa8-626f-40df-bc2e-557830e157bf

*Voz generada con [elevenlabs.io](https://elevenlabs.io).*

Prototipo de análisis de radiografías de tórax para apoyar la detección de **neumonía, COVID-19 y tuberculosis**.
Combina un ensemble de tres modelos de clasificación por enfermedad con tres visualizaciones: **Grad-CAM**
(dónde "mira" el modelo), **YOLOv8** (detección de lesiones) y **U-Net** (segmentación pulmonar).

> **Aviso importante.** Es un proyecto de investigación y portafolio. **No es un dispositivo médico, no
> está validado clínicamente y sus resultados no constituyen un diagnóstico.** Lee la sección
> [Limitaciones](#limitaciones-y-generalización) antes de interpretar cualquier resultado.

## Qué incluye

- Análisis de imágenes JPG/PNG y archivos **DICOM**, con arrastrar y soltar y radiografías de ejemplo.
- Resultado por enfermedad: probabilidad de enfermedad, confianza del veredicto y detalle por modelo
  (DenseNet121, ResNet50, EfficientNet-B0).
- Comparador interactivo entre la radiografía original y el mapa de calor Grad-CAM.
- Historial con filtros y exportación a CSV, informe en PDF por estudio, vista por paciente y dashboard.
- Flujo de validación por radiólogo (roles `admin` / `radiologist`), con registro de auditoría y
  exportación FHIR.
- Autenticación con JWT.
- Interfaz responsive, accesible con teclado, y que respeta `prefers-reduced-motion`.

## Tecnologías

| Parte | Tecnología |
| --- | --- |
| Backend | Python, FastAPI, SQLAlchemy (SQLite por defecto), PyTorch |
| Modelos | DenseNet121, ResNet50, EfficientNet-B0 (ensemble) · YOLOv8 · U-Net con ResNet34 |
| Frontend | React (Create React App), Recharts, lucide-react |
| Despliegue | Docker Compose + Nginx |

## Datasets y créditos

Los modelos se entrenaron con datasets públicos de Kaggle. Cada autor pide ser citado; revisa la
licencia exacta en la página de cada dataset antes de redistribuir imágenes.

| Dataset | Autores / fuente | Licencia |
| --- | --- | --- |
| [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | Kermany, Zhang, Goldbaum. Datos originales: [Mendeley Data, DOI 10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2). Pacientes pediátricos de 1 a 5 años | CC BY 4.0 (indicada en Mendeley Data) |
| [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | Chowdhury, Rahman, Khandakar et al. | Ver la página del dataset |
| [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) | Rahman, Khandakar et al. | Ver la página del dataset |
| [Pulmonary Chest X-Ray Abnormalities](https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities) (U-Net) | Conjuntos Montgomery County y Shenzhen de la U.S. National Library of Medicine. Las máscaras de pulmón son las de Montgomery (138 radiografías) | Datos públicos de la NLM; se pide citar la fuente |
| [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) (YOLO) | Vingroup Big Data Institute, a partir de VinDr-CXR | Solo uso no comercial y de investigación; los datos no se pueden redistribuir |

**Referencias a citar**

- D. S. Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning",
  *Cell* 172(5):1122–1131, 2018. DOI [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010).
- M. E. H. Chowdhury et al., "Can AI help in screening Viral and COVID-19 pneumonia?",
  *IEEE Access* 8:132665–132676, 2020.
- T. Rahman et al., "Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and
  Visualization", *IEEE Access* 8:191586–191601, 2020. DOI
  [10.1109/ACCESS.2020.3031384](https://doi.org/10.1109/ACCESS.2020.3031384).
- S. Jaeger et al., "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases",
  *Quantitative Imaging in Medicine and Surgery* 4(6):475–477, 2014.
- S. Jaeger et al., "Automatic tuberculosis screening using chest radiographs", *IEEE Transactions on
  Medical Imaging* 33(2):233–245, 2014. DOI [10.1109/TMI.2013.2284099](https://doi.org/10.1109/TMI.2013.2284099).
- S. Candemir et al., "Lung segmentation in chest radiographs using anatomical atlases with nonrigid
  registration", *IEEE Transactions on Medical Imaging* 33(2):577–590, 2014. DOI
  [10.1109/TMI.2013.2290491](https://doi.org/10.1109/TMI.2013.2290491).
- H. Q. Nguyen et al., "VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations",
  *Scientific Data* 9:429, 2022. DOI [10.1038/s41597-022-01498-w](https://doi.org/10.1038/s41597-022-01498-w).

Los conjuntos Montgomery y Shenzhen proceden de la National Library of Medicine (National Institutes of
Health, Bethesda, MD, USA) y del Shenzhen No.3 People's Hospital (Guangdong Medical College, Shenzhen, China).

### Imágenes de ejemplo

Las radiografías de `frontend-react2/public/samples/` son imágenes públicas de investigación, sin datos de
pacientes identificables.

| Archivo | Imagen original | Dataset | Cita |
| --- | --- | --- | --- |
| `ejemplo-1.jpg` | `Tuberculosis-100.png` | "Tuberculosis (TB) Chest X-ray Database": T. Rahman, A. Khandakar, M. E. H. Chowdhury, IEEE Dataport, 2020, DOI [10.21227/mps8-kb56](https://doi.org/10.21227/mps8-kb56) | Rahman et al., "Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization", *IEEE Access*, 2020 |
| `ejemplo-2.jpg` | `person102_bacteria_487.jpeg` | "Chest X-Ray Images (Pneumonia)": Kermany, Zhang y Goldbaum, Mendeley Data, 2018, DOI [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2), licencia CC BY 4.0 | Kermany et al., *Cell* 172(5):1122–1131, 2018, DOI [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010) |
| `ejemplo-3.jpg` | `person1950_bacteria_4881.jpeg` | Igual que `ejemplo-2.jpg` | Igual que `ejemplo-2.jpg` |

## Limitaciones y generalización

Este es el punto más importante del proyecto: **un buen resultado en el conjunto de prueba no garantiza que
el modelo funcione con radiografías de otros hospitales, equipos o pacientes.**

- **Orígenes distintos.** Cada enfermedad proviene de un dataset diferente. Un modelo puede aprender rasgos
  del *origen* de la imagen (marcas de texto, contraste, tipo de equipo, edad del paciente) en lugar de la
  enfermedad. Esto se conoce como *aprendizaje por atajos*.
- **Métricas sospechosas.** El panel de validación muestra AUC de 1.000 para COVID-19 y tuberculosis. Un
  resultado tan perfecto suele indicar sobreajuste, un conjunto de prueba pequeño o fuga de datos entre
  entrenamiento y prueba, no un modelo infalible.
- **Fuera de distribución.** Un modelo de neumonía entrenado con niños de 1 a 5 años no tiene por qué
  funcionar con adultos, y viceversa. Una radiografía sin relación con el entrenamiento recibirá igualmente
  una probabilidad, aunque no tenga sentido clínico.
- **Sin validación externa.** No se ha evaluado con datos de un hospital distinto ni con revisión de
  radiólogos.
- **Grad-CAM no es una prueba.** Muestra qué zonas influyeron en la decisión del modelo; no demuestra que sean
  zonas clínicamente relevantes.

Lecturas recomendadas sobre este problema:

- A. J. DeGrave, J. D. Janizek, S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over
  signal", *Nature Machine Intelligence* 3:610–619, 2021.
  DOI [10.1038/s42256-021-00338-7](https://doi.org/10.1038/s42256-021-00338-7).
- J. R. Zech et al., "Variable generalization performance of a deep learning model to detect pneumonia in
  chest radiographs: A cross-sectional study", *PLOS Medicine* 15(11):e1002683, 2018.

## Cómo ejecutarlo

Requisitos: Python 3.11 y Node.js 18+.

### 1. Modelos

Los modelos no están en este repositorio. Se publican en
[Hugging Face](https://huggingface.co/JRLR22/rt26-models) (unos 500 MB, licencia CC BY-NC 4.0) y se
descargan con `python download_models.py`, dentro del paso 2. El script solo baja lo que falta, usa una
versión fija del repositorio de modelos y comprueba el SHA-256 de cada archivo. Con `--verify` comprueba
también los que ya tengas.

En la [ficha de los modelos](https://huggingface.co/JRLR22/rt26-models) están los datos de entrenamiento,
las métricas y las limitaciones.

#### Detección YOLO (opcional)

Los pesos de YOLO **no se publican**: se entrenaron con los datos de la competición VinBigData, cuyas reglas
limitan su uso a fines no comerciales y de investigación. Sin ese archivo, la aplicación funciona igual y
oculta la vista YOLO. Si tienes acceso a esos datos y entrenas tu propio detector YOLOv8 con sus 14 clases,
guárdalo como `backend/models/yolov8_chest_best.pt` y se activará al reiniciar el backend.

### 2. Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_models.py   # descarga los modelos de Hugging Face
copy .env.example .env      # y define un SECRET_KEY propio
python init_db.py           # crea la base de datos y un usuario administrador inicial
$env:PYTHONUTF8=1; python main.py
```

La API queda en `http://localhost:8000` y su documentación en `http://localhost:8000/docs`.

> `init_db.py` crea el administrador con `ADMIN_EMAIL` y `ADMIN_PASSWORD` de `backend/.env`. Si no defines
> `ADMIN_PASSWORD`, genera una contraseña aleatoria y **la muestra una sola vez en la consola**: guárdala.
> Si ese usuario ya existe, no cambia su contraseña.

### 3. Frontend

```powershell
cd frontend-react2
npm install
npm start
```

Se abre en `http://localhost:3000`.

Para mostrar el botón **"Probar demo sin cuenta"**, crea `frontend-react2/.env` con una cuenta de
solo lectura (rol `viewer`). Estos valores quedan visibles en el JavaScript publicado, así que no uses
una cuenta con privilegios:

```
REACT_APP_DEMO_EMAIL=correo-de-la-cuenta-demo
REACT_APP_DEMO_PASSWORD=contraseña-de-la-cuenta-demo
```

### Con Docker

Antes, descarga los modelos en tu equipo (paso 2 hasta `python download_models.py`): el contenedor los
monta desde `backend/models/` en modo de solo lectura.

```powershell
copy backend\.env.example .env   # define SECRET_KEY, DATABASE_URL, ALLOWED_ORIGINS…
docker-compose up --build
```

El archivo `docker-compose.prod.yml` añade la configuración para producción con HTTPS.

### Asistente de chat (opcional)

La interfaz incluye un widget de chat, pero está **desactivado por defecto**: se conecta a un servicio
aparte en `http://localhost:8001` (un agente local con Ollama) que **no forma parte de este repositorio**.
El resto de la aplicación funciona normalmente sin él. Si tienes ese servicio corriendo, actívalo con
`REACT_APP_ENABLE_CHAT=true` en `frontend-react2/.env` y reinicia `npm start`.

## Estructura

```
backend/            API FastAPI, modelos de datos, servicios de inferencia
frontend-react2/    Interfaz React (la versión actual)
docker-compose*.yml Despliegue con Nginx
```

## Licencia

- **Código:** [GNU AGPL-3.0](LICENSE), la misma licencia que `ultralytics`, del que depende el backend.
  Puedes usarlo y modificarlo, pero si lo distribuyes o lo ofreces como servicio en red debes publicar
  el código fuente con la misma licencia.
- **Pesos de los modelos** (Hugging Face): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
  solo uso no comercial.
- **Datasets:** mantienen la licencia y los términos de sus autores originales.

Copyright © 2026 Raúl López.
