---
base_model: facebook/nllb-200-distilled-600M
library_name: peft
pipeline_tag: translation
tags:
- base_model:adapter:facebook/nllb-200-distilled-600M
- lora
- transformers
- translation
- bikol
---

# Model Card for Bikolé LoRA Adapter

A LoRA adapter for NLLB-200 (600M) that translates between English, Virac (Southern Catanduanes Bikol) and Pandan (Northern Catanduanes Bikol) in all six directions.

## Model Details

### Model Description

- **Developed by:** Shiela I. Trinidad and Merylle T. Rodriguez
- **Model type:** LoRA adapter (PEFT) on a sequence-to-sequence translation model
- **Language(s) (NLP):** English, Virac (Southern Bikol), Pandan (Northern Bikol)
- **License:** Not yet chosen by the authors. The base model, NLLB-200, is released under CC-BY-NC 4.0, so any use of this adapter is also subject to that license.
- **Finetuned from model:** `facebook/nllb-200-distilled-600M`

### Model Sources

- **Repository:** https://github.com/sitrinidad/bikol-nmt-capstone
- **Adapter weights:** GitHub Release v1.0.1 (`adapter_model.safetensors`, 69.3 MB)
- **Tokenizer file:** GitHub Release v1.0.2 (`tokenizer.json`)
- **Demo:** Bikolé web app in the repository (`uvicorn backend.main:app --reload`)

## Uses

### Direct Use

Translating short, everyday sentences between English, Virac and Pandan, for residents, students, teachers and local-office staff in Catanduanes. One adapter serves all six directions: English to Virac, English to Pandan, Virac to English, Pandan to English, Virac to Pandan and Pandan to Virac.

### Downstream Use

The adapter is served by the Bikolé FastAPI service, which adds sentence splitting, a confidence score and a feedback endpoint so users can report wrong translations.

### Out-of-Scope Use

Long documents, news, formal or legal text, and conversational text. The training data is short phrasebook-style sentences, so scores will not carry over. The model should not be the only source for medical, legal or safety-critical communication.

## Bias, Risks, and Limitations

- The corpus is small (1,569 aligned sentence trios) and was written by only two people, one Pandan speaker and one Virac speaker.
- NLLB-200 has no language code for Bikol, so the training borrows existing tags: `tgl_Latn` for Pandan (`bcl_North`) and `war_Latn` for Virac (`bcl_South`). English uses `eng_Latn`.
- The model over-fits: chrF++ on training sentences is 78 to 95, but only about 45 to 50 on held-out sentences.
- Translating into Bikol is harder than into English (mean test BLEU 20.4 into Bikol, 29.6 into English).
- The most common error is a fluent but wrong word (lexical error). Spelling standardization is only partial (12 hand-reviewed rules).
- Each direction has 235 test sentences with one reference each, so BLEU differences smaller than about 3 to 4 points are unreliable.
- The error categories in the report were tagged by a rule-based tool and have not been checked by a native speaker.

### Recommendations

Users should treat output as a draft and have a Bikol speaker check anything important. Wrong output can be sent through the Report Inaccurate Translation button, which saves corrections for future training.

## How to Get Started with the Model

Place `adapter_model.safetensors` and `tokenizer.json` (from the GitHub Releases above) in this folder, next to `adapter_config.json`, `bikol_meta.json` and `tokenizer_config.json`. Then:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

base = "facebook/nllb-200-distilled-600M"
adapter = "models/lora_adapter"

tok = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForSeq2SeqLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, adapter).eval()

# Bikol borrows existing NLLB tags: Virac -> war_Latn, Pandan -> tgl_Latn
tok.src_lang = "war_Latn"
inputs = tok("Dai mo na sya paghabulon.", return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        **inputs,
        forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
        num_beams=4,
        max_new_tokens=96,
    )
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

For the full service with the confidence score and latency, run the API from the repository (see the main README, Section 7).

## Training Details

### Training Data

`Corpus.csv`: original English, Virac and Pandan phrasebook-style sentences (greetings, food, health, directions, household talk) written by the two authors. The raw file had 1,597 rows; after cleaning, 1,569 aligned trios remain. The data was split 70/15/15 with a leak-proof group split (shared or near-duplicate sentences stay in the same split; seed 42). Training used 6,582 examples (1,097 sentence pairs times six directions).

### Training Procedure

#### Preprocessing

Light cleaning (whitespace, duplicate removal, optional diacritic removal) and 12 hand-reviewed spelling rules. Each source starts with its language tag and the target tag is forced as the first generated token. The source maximum length is 64 tokens.

#### Training Hyperparameters

- **Training regime:** fp16 mixed precision, AdamW, linear decay, warmup 6% of steps
- **LoRA:** rank 32, alpha 64, dropout 0.20
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`
- **Learning rate:** 2e-4
- **Batch size:** 16
- **Epochs:** up to 30 with early stopping (patience 5); best checkpoint at epoch 18, chosen by validation BLEU
- **Regularization:** label smoothing 0.15, weight decay 0.05, 10% source-token dropout
- **Seed:** 42

#### Speeds, Sizes, Times

- **Adapter size:** 69.3 MB (`adapter_model.safetensors`, fp32)
- **Mean latency on a Colab T4 GPU (beam 4):** 246 ms with the merged fp16 model, 417 ms with the fp32 model

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The held-out test split: 235 sentences per direction. It was used once, for the final numbers.

#### Factors

Results are reported for each of the six translation directions.

#### Metrics

BLEU, chrF++ and TER (SacreBLEU 2.6.0), decoded with beam 4, a maximum of 64 source tokens and 96 new tokens. Intervals are 95% percentile intervals from 300 paired bootstrap resamples.

### Results

| Direction | BLEU (zero-shot → LoRA) | chrF++ (zero-shot → LoRA) | TER (zero-shot → LoRA) |
|---|---|---|---|
| Pandan → English | 11.2 → 29.2 | 31.8 → 46.6 | 87.6 → 59.0 |
| English → Pandan | 3.5 → 21.3 | 28.2 → 46.8 | 99.2 → 63.5 |
| Virac → English | 15.7 → 30.0 | 36.6 → 47.3 | 87.1 → 56.2 |
| English → Virac | 1.8 → 19.6 | 24.8 → 45.3 | 100.7 → 65.3 |
| Virac → Pandan | 3.2 → 26.7 | 27.3 → 51.0 | 97.7 → 57.4 |
| Pandan → Virac | 1.5 → 23.9 | 23.6 → 50.0 | 103.8 → 59.1 |
| **Mean of 6** | **6.2 → 25.1** | **28.7 → 47.9** | **96.0 → 60.1** |

The adapter beat the zero-shot base model in all 18 direction-by-metric comparisons (paired bootstrap, p < 0.05).

#### Summary

Fine-tuning raised mean BLEU from 6.2 to 25.1. The largest gain is into Bikol, where the un-tuned model wrote Tagalog or Waray instead.

## Environmental Impact

- **Hardware Type:** NVIDIA T4 GPU (Google Colab)
- **Cloud Provider:** Google Colab
- **Hours used / Compute Region / Carbon Emitted:** not measured

## Technical Specifications

### Model Architecture and Objective

NLLB-200 distilled 600M (encoder-decoder Transformer) with LoRA adapters on the attention projections and feed-forward layers. The base weights stay frozen. The objective is standard cross-entropy with label smoothing for multilingual translation.

### Compute Infrastructure

#### Hardware

Google Colab T4 GPU for training and evaluation.

#### Software

PEFT 0.21.3, Transformers, PyTorch, SacreBLEU 2.6.0.

## Citation

NLLB Team. (2022). *No Language Left Behind: Scaling human-centered machine translation.* arXiv:2207.04672.

Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022.*

## Model Card Authors

Shiela I. Trinidad and Merylle T. Rodriguez

## Model Card Contact

Via the GitHub repository: https://github.com/sitrinidad/bikol-nmt-capstone

### Framework versions

- PEFT 0.21.3
