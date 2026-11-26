# Bikolé — Phase 3: Middleware API & Web Translation Interface

> “Naligis Here, Nagilis There—Bikolé Gets You Everywhere!”

English ↔ Virac (Southern Bikol) ↔ Pandan (Northern Bikol) translation with the NLLB-200 + LoRA
model trained in `Bikol_NMT_Phase1_Phase2_v4_1.ipynb`. Feedback is stored in **MySQL**.

```
bikole/
├── backend/            FastAPI middleware (Python)
│   ├── main.py         app, error handling, serves the frontend
│   ├── api.py          POST /api/v1/translate, POST /api/v1/feedback, GET /api/v1/languages, GET /api/v1/health
│   ├── schemas.py      request/response validation
│   ├── config.py       settings (.env file / environment variables)
│   ├── languages.py    language codes used by the model
│   └── nmt/            model loading, inference, confidence score, input cleaning
├── database/           MySQL feedback storage (database + table created automatically)
├── frontend/           HTML + CSS + JavaScript (Fetch API) + Lucide icons
├── models/             ← put your trained lora_adapter folder here
├── data/Corpus.csv     the parallel dataset
├── scripts/            export_feedback.py (feedback → CSV / JSONL for retraining)
├── .env.example        MySQL settings template
└── tests/              pytest tests for the API rules
```

## 1. Requirements

* **Python 3.10, 3.11 or 3.12** (3.11 recommended) — check with `python --version`
* **MySQL 8** or **MariaDB 10.4+** running on your computer — e.g. **XAMPP** (start *MySQL* in the XAMPP Control Panel),
  **MySQL Installer**, WAMP or Laragon
* **VS Code** with the Python extension
* ~4 GB free disk space (the NLLB base model is downloaded once, ≈ 2.5 GB) and 8 GB RAM
* Internet on the first run (to download the base model from Hugging Face)

## 2. Put the trained model in place

At the end of the Phase 2 notebook you downloaded `phase2_deliverables.zip`.
Unzip it and copy `phase2_outputs/multi/lora_adapter` into `bikole/models/` so that this file exists:

```
bikole/models/lora_adapter/bikol_meta.json
```

## 3. Set up MySQL

1. Start your MySQL server (XAMPP: click **Start** next to *MySQL*).
2. In the `bikole` folder, copy `.env.example` to a new file named `.env`.
3. Fill in your MySQL details. XAMPP default is user `root` with an **empty** password:

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=bikole
```

You don't need to create anything: on start the app creates the `bikole` database and the `feedback` table.
View saved feedback in phpMyAdmin (http://localhost/phpmyadmin → `bikole` → `feedback`) or with
`SELECT * FROM bikole.feedback;`.

## 4. Run it from the VS Code terminal

Open the `bikole` folder in VS Code (**File → Open Folder…**), then open a terminal (**Terminal → New Terminal**).

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env      # then edit .env if your MySQL has a password
uvicorn backend.main:app --reload
```

If PowerShell says *“running scripts is disabled”*, run this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env        # then edit .env if your MySQL has a password
uvicorn backend.main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

* The first start downloads the base model and loads it (a few minutes). The page shows
  “Loading the translation model…” and switches to “Model ready” when it can translate.
* API docs (try the endpoints): **http://127.0.0.1:8000/docs**
* Stop the server with **Ctrl + C**. Next time: activate the venv, start MySQL, run `uvicorn backend.main:app --reload`.

## 5. Run the tests

MySQL must be running (the tests use a separate `bikole_test` database that they create and drop):

```bash
python -m pytest
```

## 6. How translation works

* Every sentence is translated by the model — any English, Virac or Pandan sentence can be typed, whether or not
  it appears in `data/Corpus.csv`.
* Text with several sentences is translated sentence by sentence; line breaks are kept.

## 7. Website features

* Side-by-side translation with language dropdowns (same language and untrained directions are blocked), copy button
* Character counter (1,000 max) and **Clear** (empties the source and translated text)
* Real confidence score, inference latency and status cards
* **Translation history** — last 20 translations, saved in your browser; reuse, copy or delete them
* **Dark mode** — moon/sun button in the top bar; your choice is remembered
* **Error notifications** for failed requests, offline server, invalid input and the character limit
* **Report Inaccurate Translation** form saved to MySQL
* About section with the supported languages; works on desktop, tablet and phone

## 8. Settings (`.env` file or environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MYSQL_HOST` / `MYSQL_PORT` | `127.0.0.1` / `3306` | MySQL server |
| `MYSQL_USER` / `MYSQL_PASSWORD` | `root` / *(empty)* | MySQL login |
| `MYSQL_DATABASE` | `bikole` | Database name (created automatically) |
| `BIKOLE_MODEL_DIR` | `models/lora_adapter` | Adapter folder (or a merged model folder such as `merged_fp16`) |
| `BIKOLE_NUM_BEAMS` | `4` | Beam size. `1` = fastest (greedy), helps reach the < 300 ms target on CPU |
| `BIKOLE_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `BIKOLE_THREADS` | `0` | CPU threads for PyTorch (`0` = PyTorch default) |
| `BIKOLE_QUANTIZE` | `0` | `1` = dynamic int8 quantization on CPU (smaller/faster, slightly lower quality) |

## 9. API

Language codes are the codes the adapter was trained with: `eng` (English), `bcl_South` (Virac),
`bcl_North` (Pandan). Internally they are mapped to the NLLB tags stored in `bikol_meta.json`
(`eng_Latn`, `war_Latn`, `tgl_Latn`). Supported directions are read from the same file.

### POST /api/v1/translate

```json
{ "source_text": "Dai mo na sya paghabulon.", "source_lang": "bcl_South", "target_lang": "eng" }
```

```json
{ "status": "success", "translated_text": "…", "confidence_score": 0.87, "inference_latency_ms": 142 }
```

* `confidence_score` — geometric-mean token probability of the generated translation (0–1), computed by the model.
* `inference_latency_ms` — measured time for tokenising, generating and decoding.
* Errors return `{ "status": "failed", "error": "…" }` with HTTP 400 (unsupported direction), 422 (invalid input,
  e.g. same source and target language or empty text), 503 (model still loading / not found) or 500.

### POST /api/v1/feedback

```json
{
  "source_text": "Dai mo na sya paghabulon.",
  "source_lang": "bcl_South",
  "target_lang": "eng",
  "generated_translation": "Don't chase him.",
  "corrected_translation": "Don't chase after him/her anymore.",
  "explanation": "optional"
}
```

Saved in the `feedback` table of the MySQL database. Returns 503 with a clear message when MySQL is not reachable.
Export it with `python scripts/export_feedback.py`.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `uvicorn` is not recognized | The venv is not active — run the activate command again, or use `python -m uvicorn backend.main:app --reload` |
| `ModuleNotFoundError: backend` | Run the command from inside the `bikole` folder |
| “Cannot connect to MySQL at 127.0.0.1:3306” | Start MySQL (XAMPP Control Panel → MySQL → Start) |
| “MySQL refused user 'root'” | Put the right password in `.env` (`MYSQL_PASSWORD=...`) and restart uvicorn |
| Page says “Model failed to load” | Read the error in the terminal. Usually `models/lora_adapter` is missing or incomplete |
| Translation is slow | Set `BIKOLE_NUM_BEAMS=1` in `.env`; use a GPU if you have one (CUDA build of PyTorch from pytorch.org) |
| Port 8000 already in use | `uvicorn backend.main:app --reload --port 8001` |

