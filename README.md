# Bikolé: Low-Resource Neural Machine Translation for Bikol Dialects

*"Naligis Here, Nagilis There—Bikolé Gets You Everywhere!"*

CSTRACKC2 (Data Mining) capstone. English ↔ Virac (Southern Catanduanes Bikol) ↔ Pandan (Northern Catanduanes Bikol) translation, using `facebook/nllb-200-distilled-600M` fine-tuned with one LoRA adapter for six directions. The project has four parts: the corpus, the training script, the evaluation, and a web app with an API.

## 0. Overview and order of steps

1. **Data** — `data/` holds the cleaned corpus and the 70/15/15 split (seed 42, no leakage).
2. **Train** — run `train.py` (Section 2). Needs a GPU. Tested on a Colab T4.
3. **Evaluate** — run `notebooks/Bikol_NMT_Phase4_Evaluation.ipynb`. Metric tables are saved in `results/`.
4. **Serve** — start the API and web page (Sections 4–7). Needs MySQL and the trained adapter (Section 5).

### Repository structure

```
bikol-nmt-capstone/
├── README.md
├── train.py            LoRA fine-tuning script (Section 2)
├── main.py             API entry point (same app as backend/main.py)
├── requirements.txt    Python packages
├── .env.example        MySQL settings template
├── backend/            FastAPI middleware
│   ├── main.py         app, error handling, serves the frontend
│   ├── api.py          POST /api/v1/translate, POST /api/v1/feedback, GET /api/v1/languages, GET /api/v1/health
│   ├── schemas.py      request/response validation
│   ├── config.py       settings (.env file / environment variables)
│   ├── languages.py    language codes used by the model
│   └── nmt/            model loading, inference, confidence score, input cleaning
├── database/           MySQL feedback storage (database and table created automatically)
├── frontend/           HTML + CSS + JavaScript (Fetch API) + Lucide icons
├── models/lora_adapter/  adapter_config.json, bikol_meta.json, tokenizer_config.json, model card
│                         (adapter_model.safetensors and tokenizer.json come from the GitHub Releases, Section 5)
├── data/               corpus and split files (Section 1)
├── results/            metric tables, latency log, error analysis, figures
├── notebooks/          Phase 1+2 (corpus, training) and Phase 4 (evaluation) notebooks
├── scripts/            export_feedback.py (feedback → CSV / JSONL)
└── tests/              pytest tests for the API rules
```

## 1. Data

| File | What it is |
|---|---|
| `data/Corpus.csv` | raw parallel corpus (English, Virac, Pandan) written by the team |
| `data/corpus_clean_all.csv` | cleaned corpus, 1,569 aligned trios, with a `split` column |
| `data/train.csv`, `val.csv`, `test.csv` (and `.json`) | 70/15/15 split files (1,098 / 236 / 235 rows) |
| `data/directional/` | one JSONL file per split and direction; `train.py` reads these |

The corpus has no personal information. The cleaning steps, the leakage check and the vocabulary analysis are in the Phase 1 notebook (notebooks/Bikol_NMT_Phase1_Phase2_v4.ipynb) and in the report.

## 2. Train the model (`train.py`)

Needs a GPU. On Google Colab choose **Runtime → Change runtime type → T4 GPU**, then run:

```
!git clone https://github.com/sitrinidad/bikol-nmt-capstone
%cd bikol-nmt-capstone
!pip install -q transformers peft accelerate sacrebleu sentencepiece pandas matplotlib
!pip -q uninstall -y torchao
!python train.py
```

If you run this a second time in the same Colab session, the clone fails because the folder already exists. Delete it first with `%cd /content` and `!rm -rf bikol-nmt-capstone`, or start a new runtime.

- Settings: LoRA r=32, alpha=64, dropout 0.2, learning rate 2e-4, seed 42, early stopping on validation BLEU (patience 5). They can be changed with command-line flags, for example `python train.py --r 16 --alpha 32`.
- The script reads only the train and validation files. The test set is not used in training.
- The validation BLEU printed during training (and saved as `best_val_BLEU` in `bikol_meta.json`) uses greedy decoding, so it can differ from the beam-4 scores in the report.
- Output: `models/lora_adapter/` (adapter, tokenizer, `bikol_meta.json`), `results/loss_history.csv`, `results/figures/loss_curves.png`.

## 3. Evaluate

Open `notebooks/Bikol_NMT_Phase4_Evaluation.ipynb` in Colab, upload the adapter folder when asked, and run all cells. It scores the zero-shot model and the LoRA model on the test set (BLEU, chrF++, TER), measures latency, and writes the error-analysis files into `results/`.

Main files in `results/`: `benchmark_table.csv`, `test_metrics_by_direction.csv`, `latency_by_length.csv`, `error_analysis_final.csv`, and `figures/`.

## 4. Web app and API: requirements

- Python 3.10, 3.11 or 3.12 (3.11 recommended). Check with `python --version`
- MySQL 8 or MariaDB 10.4+ running on your computer, for example XAMPP (start MySQL in the XAMPP Control Panel), MySQL Installer, WAMP or Laragon
- VS Code with the Python extension
- About 4 GB free disk space (the NLLB base model is downloaded once, about 2.5 GB) and 8 GB RAM
- Internet on the first run (to download the base model from Hugging Face)

## 5. Put the trained model in place

The two large adapter files are not stored in the repository folder. Download them from the GitHub Releases and put them in `models/lora_adapter/`, next to the files that are already there:

| File | Download from |
|---|---|
| `adapter_model.safetensors` (69.3 MB) | [Release v1.0.1](https://github.com/sitrinidad/bikol-nmt-capstone/releases/tag/v1.0.1) |
| `tokenizer.json` | [Release v1.0.2](https://github.com/sitrinidad/bikol-nmt-capstone/releases/tag/v1.0.2) |

After this step, `models/lora_adapter/` must contain `adapter_config.json`, `adapter_model.safetensors`, `bikol_meta.json`, `tokenizer.json` and `tokenizer_config.json`.

If you trained the model yourself, `train.py` already saves all of these there, so you can skip the download.

**Faster model (recommended for the 300 ms target).** The default setup uses fp32 and is slower (mean 417 ms in our test). The merged fp16 model had a mean of 246 ms on a Colab T4. To use it, put the merged fp16 model folder in `models/` and set `BIKOLE_MODEL_DIR` in `.env` (Section 8).

## 6. Set up MySQL

1. Start your MySQL server (XAMPP: click **Start** next to MySQL).
2. In the project folder, copy `.env.example` to a new file named `.env`.
3. Fill in your MySQL details. The XAMPP default is user `root` with an empty password:

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=bikole
```

You do not need to create anything. When the app starts it creates the `bikole` database and the `feedback` table. To see saved feedback, use phpMyAdmin (`http://localhost/phpmyadmin` → `bikole` → `feedback`) or run `SELECT * FROM bikole.feedback;`. Do not commit your `.env` file.

## 7. Run the app

Open the project folder in VS Code (**File → Open Folder…**), then open a terminal (**Terminal → New Terminal**).

**Windows (PowerShell)**

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env      # then edit .env if your MySQL has a password
uvicorn backend.main:app --reload
```

If PowerShell says "running scripts is disabled", run this once and then activate again:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**macOS / Linux**

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env        # then edit .env if your MySQL has a password
uvicorn backend.main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

- The first start downloads the base model and loads it (a few minutes). The page shows "Loading the translation model…" and changes to "Model ready" when it can translate.
- API docs (try the endpoints): `http://127.0.0.1:8000/docs`
- Stop the server with `Ctrl + C`. Next time: activate the venv, start MySQL, run `uvicorn backend.main:app --reload`.

### Run the tests

MySQL must be running. The tests use a separate `bikole_test` database that they create and drop.

```
python -m pytest
```

### How translation works

- Every sentence is translated by the model. Any English, Virac or Pandan sentence can be typed, whether or not it appears in the corpus.
- Text with several sentences is translated sentence by sentence. Line breaks are kept.
- NLLB-200 has no Bikol language code, so each language borrows its own existing tag (`eng_Latn` for English, `tgl_Latn` for Pandan, `war_Latn` for Virac). The adapter learns to give these tags a new meaning. This is a technical device, not a claim that Pandan is Tagalog or Virac is Waray.

### Website features

- Side-by-side translation with language dropdowns (same language and untrained directions are blocked), copy button
- Character counter (1,000 max) and Clear (empties the source and translated text)
- Real confidence score, inference latency and status cards
- Translation history: last 20 translations, saved in your browser; reuse, copy or delete them
- Dark mode: moon/sun button in the top bar; your choice is remembered
- Error notifications for failed requests, offline server, invalid input and the character limit
- Report Inaccurate Translation form saved to MySQL
- About section with the supported languages; works on desktop, tablet and phone

## 8. Settings (`.env` file or environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MYSQL_HOST` / `MYSQL_PORT` | `127.0.0.1` / `3306` | MySQL server |
| `MYSQL_USER` / `MYSQL_PASSWORD` | `root` / (empty) | MySQL login |
| `MYSQL_DATABASE` | `bikole` | Database name (created automatically) |
| `BIKOLE_MODEL_DIR` | `models/lora_adapter` | Adapter folder, or a merged model folder such as `merged_fp16` |
| `BIKOLE_NUM_BEAMS` | `4` | Beam size. 1 = fastest (greedy); helps reach the < 300 ms target on CPU |
| `BIKOLE_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `BIKOLE_THREADS` | `0` | CPU threads for PyTorch (0 = PyTorch default) |
| `BIKOLE_QUANTIZE` | `0` | 1 = dynamic int8 quantization on CPU (smaller and faster, slightly lower quality) |

## 9. API

Language codes are the codes the adapter was trained with: `eng` (English), `bcl_South` (Virac), `bcl_North` (Pandan). Internally they are mapped to the NLLB tags stored in `bikol_meta.json` (`eng_Latn`, `war_Latn`, `tgl_Latn`). Supported directions are read from the same file.

### `POST /api/v1/translate`

Request:

```json
{ "source_text": "Dai mo na sya paghabulon.", "source_lang": "bcl_South", "target_lang": "eng" }
```

Response (from our running server):

```json
{
  "status": "success",
  "translated_text": "Don't chase after him/her anymore.",
  "confidence_score": 0.8156,
  "inference_latency_ms": 2717
}
```

- `confidence_score`: geometric-mean token probability of the generated translation (0–1).
- `inference_latency_ms`: measured time for tokenising, generating and decoding. This example was timed on a CPU, so it is much slower than the Colab T4 figures in Section 5.
- Errors return `{ "status": "failed", "error": "…" }` with HTTP 400 (unsupported direction), 422 (invalid input, for example same source and target language or empty text), 503 (model still loading or not found) or 500.

### `POST /api/v1/feedback`

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

Saved in the `feedback` table of the MySQL database. Returns 503 with a clear message when MySQL is not reachable. Export it with `python scripts/export_feedback.py`.

Other endpoints: `GET /api/v1/languages`, `GET /api/v1/health`.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `uvicorn` is not recognized | The venv is not active. Run the activate command again, or use `python -m uvicorn backend.main:app --reload` |
| `ModuleNotFoundError: backend` | Run the command from inside the project folder |
| "Cannot connect to MySQL at 127.0.0.1:3306" | Start MySQL (XAMPP Control Panel → MySQL → Start) |
| "MySQL refused user 'root'" | Put the right password in `.env` (`MYSQL_PASSWORD=...`) and restart uvicorn |
| Page says "Model failed to load" | Read the error in the terminal. Usually `adapter_model.safetensors` or `tokenizer.json` was not downloaded into `models/lora_adapter/` (Section 5) |
| Translation is slow | Use the merged fp16 model on a GPU, or set `BIKOLE_NUM_BEAMS=1` in `.env` (CUDA build of PyTorch from pytorch.org) |
| Port 8000 already in use | `uvicorn backend.main:app --reload --port 8001` |
| `git clone` says the folder already exists (Colab) | Delete the old folder with `!rm -rf bikol-nmt-capstone`, or start a new runtime |

## 11. Project info

- Course: CSTRACKC2 – Data Mining, AY 2026–2027, first semester
- Authors: Shiela I. Trinidad, Merylle T. Rodriguez
- Base model: `facebook/nllb-200-distilled-600M` (NLLB Team, 2022); LoRA via PEFT
- Random seed: 42 (split, training and evaluation)
- Tools: Google Colab (training and evaluation), Replit (web interface design help), and Claude and ChatGPT (see the AI tool disclosure in the report)
