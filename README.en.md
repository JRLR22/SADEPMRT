# rt26 · AI-assisted chest X-ray analysis

[![CI](https://github.com/JRLR22/SADEPMRT/actions/workflows/ci.yml/badge.svg)](https://github.com/JRLR22/SADEPMRT/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![Models on Hugging Face](https://img.shields.io/badge/models-Hugging%20Face-yellow)](https://huggingface.co/JRLR22/rt26-models)

[Versión en español](README.md)

https://github.com/user-attachments/assets/a1798fa8-626f-40df-bc2e-557830e157bf

*Voice generated with [elevenlabs.io](https://elevenlabs.io).*

Prototype that analyzes chest X-rays to support the detection of **pneumonia, COVID-19 and tuberculosis**.
It combines an ensemble of three classifiers per disease with three visualizations: **Grad-CAM** (where the
model "looks"), **YOLOv8** (lesion detection) and **U-Net** (lung segmentation). The user interface is in
Spanish.

> **Important.** This is a research and portfolio project. **It is not a medical device, it has not been
> clinically validated and its output is not a diagnosis.** Read the [Limitations](#limitations-and-generalization)
> section before interpreting any result.

## Features

- Analysis of JPG/PNG images and **DICOM** files, with drag and drop and sample X-rays.
- Per-disease results: disease probability, confidence of the verdict and per-model detail
  (DenseNet121, ResNet50, EfficientNet-B0).
- Interactive slider comparing the original X-ray with the Grad-CAM heatmap.
- History with filters and CSV export, PDF report per study, per-patient view and dashboard.
- Radiologist validation workflow (`admin` / `radiologist` roles), with audit log and FHIR export.
- JWT authentication.
- Responsive, keyboard-accessible interface that respects `prefers-reduced-motion`.

## Tech stack

| Part | Technology |
| --- | --- |
| Backend | Python, FastAPI, SQLAlchemy (SQLite by default), PyTorch |
| Models | DenseNet121, ResNet50, EfficientNet-B0 (ensemble) · YOLOv8 · U-Net with ResNet34 |
| Frontend | React (Create React App), Recharts, lucide-react |
| Deployment | Docker Compose + Nginx |
| Quality | pytest, Jest, GitHub Actions |

## Datasets and credits

The models were trained on public Kaggle datasets. Each author asks to be cited; check the exact license on
each dataset page before redistributing images.

| Dataset | Authors / source | License |
| --- | --- | --- |
| [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | Kermany, Zhang, Goldbaum. Original data: [Mendeley Data, DOI 10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2). Pediatric patients aged 1 to 5 | CC BY 4.0 (stated on Mendeley Data) |
| [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | Chowdhury, Rahman, Khandakar et al. | See the dataset page |
| [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) | Rahman, Khandakar et al. | See the dataset page |
| [Pulmonary Chest X-Ray Abnormalities](https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities) (U-Net) | Montgomery County and Shenzhen sets from the U.S. National Library of Medicine. The lung masks are the Montgomery ones (138 X-rays) | Public NLM data; citing the source is requested |
| [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) (YOLO) | Vingroup Big Data Institute, based on VinDr-CXR | Non-commercial and research use only; the data may not be redistributed |

**References to cite**

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

The Montgomery and Shenzhen sets come from the National Library of Medicine (National Institutes of Health,
Bethesda, MD, USA) and the Shenzhen No.3 People's Hospital (Guangdong Medical College, Shenzhen, China).

### Sample images

The X-rays in `frontend-react2/public/samples/` are public research images with no identifiable patient
data.

| File | Original image | Dataset | Citation |
| --- | --- | --- | --- |
| `ejemplo-1.jpg` | `Tuberculosis-100.png` | "Tuberculosis (TB) Chest X-ray Database": T. Rahman, A. Khandakar, M. E. H. Chowdhury, IEEE Dataport, 2020, DOI [10.21227/mps8-kb56](https://doi.org/10.21227/mps8-kb56) | Rahman et al., "Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization", *IEEE Access*, 2020 |
| `ejemplo-2.jpg` | `person102_bacteria_487.jpeg` | "Chest X-Ray Images (Pneumonia)": Kermany, Zhang and Goldbaum, Mendeley Data, 2018, DOI [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2), CC BY 4.0 license | Kermany et al., *Cell* 172(5):1122–1131, 2018, DOI [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010) |
| `ejemplo-3.jpg` | `person1950_bacteria_4881.jpeg` | Same as `ejemplo-2.jpg` | Same as `ejemplo-2.jpg` |

## Limitations and generalization

This is the most important part of the project: **a good score on the test set does not guarantee that the
model works on X-rays from other hospitals, machines or patients.**

- **Different sources.** Each disease comes from a different dataset. A model can learn features of the
  image *source* (text markers, contrast, equipment, patient age) instead of the disease. This is known as
  *shortcut learning*.
- **Suspicious metrics.** The validation panel shows an AUC of 1.000 for COVID-19 and tuberculosis. Such a
  perfect result usually points to overfitting, a small test set or leakage between training and test data,
  not to an infallible model.
- **Out of distribution.** A pneumonia model trained on children aged 1 to 5 has no reason to work on
  adults, and vice versa. An X-ray unrelated to the training data still gets a probability, even if it
  makes no clinical sense.
- **No external validation.** It has not been evaluated on data from another hospital or reviewed by
  radiologists.
- **Grad-CAM is not proof.** It shows which regions influenced the model's decision; it does not show that
  those regions are clinically relevant.

Recommended reading on this problem:

- A. J. DeGrave, J. D. Janizek, S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over
  signal", *Nature Machine Intelligence* 3:610–619, 2021.
  DOI [10.1038/s42256-021-00338-7](https://doi.org/10.1038/s42256-021-00338-7).
- J. R. Zech et al., "Variable generalization performance of a deep learning model to detect pneumonia in
  chest radiographs: A cross-sectional study", *PLOS Medicine* 15(11):e1002683, 2018.

## Getting started

Requirements: Python 3.11 and Node.js 18+.

### 1. Models

The models are not stored in this repository. They are published on
[Hugging Face](https://huggingface.co/JRLR22/rt26-models) (about 500 MB, CC BY-NC 4.0 license) and are
downloaded with `python download_models.py` during step 2. The script only fetches what is missing, pins a
fixed revision of the model repository and checks the SHA-256 of every file. With `--verify` it also checks
the files you already have.

The [model card](https://huggingface.co/JRLR22/rt26-models) describes the training data, metrics and
limitations.

#### YOLO detection (optional)

The YOLO weights are **not published**: they were trained on the VinBigData competition data, whose rules
restrict its use to non-commercial and research purposes. Without that file the application works the same
and hides the YOLO view. If you have access to that data and train your own YOLOv8 detector with its 14
classes, save it as `backend/models/yolov8_chest_best.pt` and it will be enabled when the backend restarts.

### 2. Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_models.py   # downloads the models from Hugging Face
copy .env.example .env      # and set your own SECRET_KEY
python init_db.py           # creates the database and an initial admin user
$env:PYTHONUTF8=1; python main.py
```

The API runs at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

> `init_db.py` creates the admin user from `ADMIN_EMAIL` and `ADMIN_PASSWORD` in `backend/.env`. If
> `ADMIN_PASSWORD` is not set, it generates a random password and **prints it to the console only once**:
> save it. If that user already exists, its password is left unchanged.

### 3. Frontend

```powershell
cd frontend-react2
npm install
npm start
```

It opens at `http://localhost:3000`.

To show the **"Probar demo sin cuenta"** (try the demo) button, create `frontend-react2/.env` with a
read-only account (`viewer` role). These values end up visible in the published JavaScript, so do not use
a privileged account:

```
REACT_APP_DEMO_EMAIL=demo-account-email
REACT_APP_DEMO_PASSWORD=demo-account-password
```

### With Docker

First download the models on your machine (step 2 up to `python download_models.py`): the container mounts
them read-only from `backend/models/`.

```powershell
copy backend\.env.example .env   # set SECRET_KEY, DATABASE_URL, ALLOWED_ORIGINS…
docker-compose up --build
```

`docker-compose.prod.yml` adds the production setup with HTTPS.

### Chat assistant (optional)

The interface includes a chat widget, but it is **disabled by default**: it talks to a separate service at
`http://localhost:8001` (a local agent running on Ollama) that is **not part of this repository**. The rest
of the application works normally without it. If that service is running, enable it with
`REACT_APP_ENABLE_CHAT=true` in `frontend-react2/.env` and restart `npm start`.

## Tests

```powershell
cd backend
pip install -r requirements-dev.txt
pytest
```

The backend tests cover configuration, security, file validation, inference, every API endpoint (role
permissions, audit log, rate limiting, PDF, DICOM), `init_db.py` and `download_models.py`. They use a
temporary database and working folder plus randomly initialized models, so they never touch your data. One
extra test starts the API with the real models and is skipped if they have not been downloaded.

In the frontend, `npm test` runs a smoke test of the login screen.

GitHub Actions runs both on every push to `main` and on every pull request, with the models downloaded from
Hugging Face.

## Project structure

```
backend/
  main.py             startup, middleware and route registration
  app/api/            endpoints: auth, analysis, studies, reports, dashboard, health
  app/core/           security (JWT), file validation and rate limiting
  app/services/       inference (ensemble, Grad-CAM, YOLO, U-Net), records and storage
  app/models/         database models (SQLAlchemy)
  tests/              pytest test suite
  download_models.py  downloads the models from Hugging Face
frontend-react2/      React interface
docker-compose*.yml   deployment with Nginx
```

## License

- **Code:** [GNU AGPL-3.0](LICENSE), the same license as `ultralytics`, which the backend depends on. You
  may use and modify it, but if you distribute it or offer it as a network service you must publish the
  source code under the same license.
- **Model weights** (Hugging Face): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
  non-commercial use only.
- **Datasets:** keep the licenses and terms of their original authors.

Copyright © 2026 Raúl López.
