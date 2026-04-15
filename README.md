# KarierAI

`KarierAI` adalah aplikasi **KarierAI** untuk membantu:
- chat seputar lowongan kerja
- analisis CV dari **teks**, **PDF**, **PDF hasil scan**, dan **gambar**
- rekomendasi pekerjaan berdasarkan CV
- konsultasi gap skill untuk target role tertentu
- query data lowongan dengan jalur **text-to-SQL** yang lebih fleksibel namun tetap aman

Project ini memakai:
- **FastAPI** sebagai backend API
- **Streamlit** sebagai frontend sederhana
- **SQLite** untuk query terstruktur
- **Qdrant** untuk semantic retrieval (opsional)
- **OpenAI API** untuk kemampuan AI yang lebih lengkap (opsional)
- **Tesseract OCR** untuk membaca CV hasil scan atau gambar

> **Catatan penting**

> - OCR untuk PDF scan dan gambar membutuhkan **Tesseract OCR** terpasang di sistem.

---

## Fitur Utama

### 1. Chat lowongan kerja
User bisa bertanya tentang lowongan, statistik sederhana, atau pencarian berdasarkan role/lokasi.

### 2. Analisis CV
CV bisa dianalisis dari:
- teks langsung
- file PDF berbasis teks
- file PDF hasil scan / image-only
- file gambar: `png`, `jpg`, `jpeg`, `webp`, `bmp`, `tif`, `tiff`

Sistem akan mengekstrak informasi seperti:
- skill yang terdeteksi
- role yang paling mungkin cocok
- indikasi pengalaman kerja
- ringkasan profil

### 3. Rekomendasi lowongan
CV dicocokkan dengan dataset lowongan dan menghasilkan rekomendasi job yang paling relevan.

### 4. Career consultation
Menganalisis gap skill terhadap role target, misalnya `Data Scientist`, `Data Analyst`, atau `HR Manager`.

### 5. Text-to-SQL yang lebih fleksibel
Project ini sudah mendukung jalur query berbasis bahasa natural ke SQL dengan cakupan yang lebih fleksibel untuk kebutuhan seperti:
- `count`
- `distinct`
- `group by`
- agregasi per lokasi / perusahaan / work type / role
- metrik salary melalui fungsi aman seperti `salary_min`, `salary_max`, dan `salary_mid`

Jika `OPENAI_API_KEY` tersedia, backend dapat mencoba membuat SQL **read-only** yang tetap divalidasi sebelum dijalankan.

---

## Struktur Project

```text
KarierAI/
├── dataset/
│   ├── jobs.jsonl
│   └── jobs.db
├── script/
│   ├── init_sqlite.py
│   └── run_ingestion.py
├── src/
│   └── karierai/
│       ├── __init__.py
│       ├── config.py
│       ├── telemetry.py
│       ├── prompts.py
│       ├── models.py
│       ├── database.py
│       ├── ingestion.py
│       ├── services.py
│       ├── tools.py
│       ├── agent.py
│       ├── server.py
│       └── simulation.py
├── tests/
│   ├── conftest.py
│   ├── test_chat.py
│   ├── test_features.py
│   └── test_health.py
├── deployment_gcp.md
├── Dockerfile
├── flow.png
├── pyproject.toml
└── README.md
```

---

## Requirements

### Minimum
- **Python 3.11+**
- **Poetry**
- koneksi internet jika ingin memakai OpenAI / Qdrant cloud

### Tambahan untuk OCR
Agar CV scan dan gambar bisa dibaca, install **Tesseract OCR** di sistem.

#### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ind tesseract-ocr-eng
```

#### macOS (Homebrew)
```bash
brew install tesseract
```

#### Windows
Install Tesseract OCR lalu pastikan binary `tesseract` masuk ke `PATH`.

---

## Konfigurasi Environment

File `.env` sudah tersedia di project ini.
Cukup **edit file `.env`** sesuai kebutuhan.

Contoh konfigurasi minimum:

```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
API_URL=http://localhost:8080
API_PORT=8080
```

Catatan:
- `OPENAI_API_KEY` **opsional**, tetapi dibutuhkan untuk fitur AI penuh dan jalur text-to-SQL berbasis LLM.
- Jika tidak diisi, beberapa fitur tetap berjalan dengan fallback lokal.
- `QDRANT_URL` dan `QDRANT_API_KEY` **opsional**.

---

## Menjalankan Project Secara Lokal

### 1. Install Poetry
Jika Poetry belum terpasang:

```bash
pip install poetry
```

atau:

```bash
python -m pip install poetry
```

### 2. Install dependency Python
```bash
poetry install
```

### 3. Inisialisasi database SQLite
```bash
PYTHONPATH=src poetry run python script/init_sqlite.py
```

### 4. Masukkan dataset ke database
```bash
PYTHONPATH=src poetry run python script/run_ingestion.py
```

### 5. Jalankan backend API
```bash
PYTHONPATH=src poetry run uvicorn karierai.server:app --host 0.0.0.0 --port 8080 --reload
```

Backend akan aktif di:

```text
http://localhost:8080
```

### 6. Jalankan frontend Streamlit
Buka terminal baru, lalu jalankan:

```bash
PYTHONPATH=src poetry run streamlit run src/karierai/simulation.py --server.port 8501
```

Frontend akan aktif di:

```text
http://localhost:8501
```

---

## Menjalankan Test

```bash
PYTHONPATH=src poetry run pytest
```

---

## Menjalankan dengan Docker

Project ini memiliki `Dockerfile`, tetapi sebelum dipakai perhatikan dua hal berikut:

 untuk OCR di container, image juga perlu memiliki **Tesseract OCR** terpasang

### Build image
```bash
docker build -t karierai .
```

### Init SQLite
#### Linux/macOS
```bash
docker run --rm --env-file .env -v "$(pwd)/dataset:/app/dataset" karierai sh -c "PYTHONPATH=src python script/init_sqlite.py"
```

#### Windows PowerShell
```powershell
docker run --rm --env-file .env -v "${PWD}/dataset:/app/dataset" karierai sh -c "PYTHONPATH=src python script/init_sqlite.py"
```

### Ingestion dataset
#### Linux/macOS
```bash
docker run --rm --env-file .env -v "$(pwd)/dataset:/app/dataset" karierai sh -c "PYTHONPATH=src python script/run_ingestion.py"
```

#### Windows PowerShell
```powershell
docker run --rm --env-file .env -v "${PWD}/dataset:/app/dataset" karierai sh -c "PYTHONPATH=src python script/run_ingestion.py"
```

### Jalankan API
#### Linux/macOS
```bash
docker run --rm --env-file .env -v "$(pwd)/dataset:/app/dataset" -p 8080:8080 karierai
```

#### Windows PowerShell
```powershell
docker run --rm --env-file .env -v "${PWD}/dataset:/app/dataset" -p 8080:8080 karierai
```

Catatan:
- Docker image ini default-nya menjalankan **backend API** saja
- Streamlit belum dijalankan otomatis dari container ini

---

## Endpoint API

### Health & readiness
- `GET /health`
- `GET /ready`

### Data
- `POST /ingest`

### Chat
- `POST /chat`

### CV analysis
- `POST /cv/analyze` → analisis dari teks
- `POST /cv/analyze-file` → analisis dari file PDF / image

### Recommendation
- `POST /recommend` → rekomendasi dari teks CV
- `POST /recommend-file` → rekomendasi dari file PDF / image

### Consultation
- `POST /consult` → konsultasi dari teks CV
- `POST /consult-file` → konsultasi dari file PDF / image

### Prompt preview
- `GET /prompts/{prompt_name}`

---

## Format File CV yang Didukung

### Input teks
- teks CV yang langsung dikirim ke body request

### Input file
- `pdf`
- `png`
- `jpg`
- `jpeg`
- `webp`
- `bmp`
- `tif`
- `tiff`

---

## Contoh Request

### Health Check
```bash
curl http://localhost:8080/health
```

### Chat
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Cari lowongan data analyst di Jakarta","history":""}'
```

### Analisis CV dari teks
```bash
curl -X POST http://localhost:8080/cv/analyze \
  -H "Content-Type: application/json" \
  -d '{"cv_text":"Data analyst with 3 years experience using SQL, Python, Tableau, and Power BI."}'
```

### Analisis CV dari PDF / image
```bash
curl -X POST http://localhost:8080/cv/analyze-file \
  -F "file=@cv.pdf"
```

### Rekomendasi pekerjaan dari PDF / image
```bash
curl -X POST http://localhost:8080/recommend-file \
  -F "file=@cv_scan.pdf" \
  -F "top_k=5"
```

### Konsultasi gap skill dari PDF / image
```bash
curl -X POST http://localhost:8080/consult-file \
  -F "file=@cv_image.png" \
  -F "target_role=Data Scientist"
```

---

## Alur Menjalankan Project

Urutan yang disarankan:

1. edit `.env`
2. install dependency Python
3. install **Tesseract OCR**
4. jalankan `init_sqlite.py`
5. jalankan `run_ingestion.py`
6. jalankan FastAPI
7. jalankan Streamlit
8. coba endpoint atau UI

---

## Troubleshooting

### `poetry: command not found`
Install Poetry dulu:

```bash
pip install poetry
```

### `ModuleNotFoundError: karierai`
Pastikan command dijalankan dengan `PYTHONPATH=src`.

### OCR tidak jalan / file scan gagal dibaca
Periksa:
- apakah **Tesseract OCR** sudah terpasang
- apakah binary `tesseract` ada di `PATH`
- apakah file scan cukup jelas dan resolusinya memadai

### Docker build gagal
Periksa apakah `Dockerfile` masih berisi `COPY data ./data` padahal folder `data/` tidak ada.

### Hasil analisis CV dari scan kurang akurat
OCR dipengaruhi oleh kualitas file. Gunakan scan yang:
- tajam
- kontras jelas
- tidak miring terlalu parah
- tidak blur

### Streamlit tidak bisa terhubung ke backend
Pastikan backend FastAPI sudah hidup di:

```text
http://localhost:8080
```

---

## Ringkasan

Project ini mendukung:
- chat lowongan kerja
- analisis CV dari teks
- analisis CV dari **PDF berbasis teks**
- analisis CV dari **PDF hasil scan**
- analisis CV dari **gambar**
- rekomendasi pekerjaan
- konsultasi gap skill
- jalur **text-to-SQL** yang lebih fleksibel dan tetap aman

Jika ingin hasil OCR optimal, pastikan **Tesseract OCR** sudah terpasang dengan language pack yang sesuai.
