"""Loads the fine-tuned NLLB-200 + LoRA model and runs real translations.

Heavy libraries (torch / transformers / peft) are imported lazily inside `load()`,
so the API, the database and the tests work even before the model is installed.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..languages import DEFAULT_CODE_MAP, DEFAULT_DIRECTIONS
from .preprocess import normalize_source, split_segments

log = logging.getLogger("bikole.nmt")

DEFAULT_BASE_MODEL = "facebook/nllb-200-distilled-600M"


class EngineState(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ModelNotReady(RuntimeError):
    pass


class UnsupportedDirection(ValueError):
    pass


@dataclass
class TranslationResult:
    translated_text: str
    confidence_score: float | None   # geometric-mean token probability of the generated output (0–1)
    inference_latency_ms: int        # measured wall time: tokenise + generate + decode


class NMTEngine:
    def __init__(self, model_dir: Path, device: str = "auto", num_beams: int = 4, quantize_cpu: bool = False,
                 cpu_threads: int = 0):
        self.model_dir = Path(model_dir)
        self.requested_device = device
        self.num_beams = max(1, int(num_beams))
        self.quantize_cpu = quantize_cpu
        self.cpu_threads = cpu_threads
        self.state = EngineState.NOT_LOADED
        self.error: str | None = None
        self.device: str | None = None
        self.base_model: str | None = None
        self.model_version: str | None = None
        self.code_map: dict[str, str] = dict(DEFAULT_CODE_MAP)
        self.directions: list[tuple[str, str]] = list(DEFAULT_DIRECTIONS)
        self.max_len = 64
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        self.state, self.error = EngineState.LOADING, None
        try:
            self._load()
            self.state = EngineState.READY
            log.info("Model ready on %s (%s)", self.device, self.model_version)
        except Exception as exc:  # keep the API alive and report the reason
            self.state, self.error = EngineState.ERROR, f"{type(exc).__name__}: {exc}"
            log.exception("Model failed to load")

    def load_in_background(self) -> threading.Thread:
        self.state = EngineState.LOADING
        t = threading.Thread(target=self.load, name="bikole-model-loader", daemon=True)
        t.start()
        return t

    def _load(self) -> None:
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Model folder not found: {self.model_dir}. Copy 'lora_adapter' from phase2_deliverables.zip "
                "into bikole/models/ or set BIKOLE_MODEL_DIR."
            )
        import torch
        import transformers
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        transformers.logging.set_verbosity_error()
        if self.cpu_threads > 0:
            torch.set_num_threads(self.cpu_threads)
        self._torch = torch
        meta_path = self.model_dir / "bikol_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.code_map = dict(meta.get("language_code_map") or DEFAULT_CODE_MAP)
        if meta.get("directions"):
            self.directions = [tuple(d) for d in meta["directions"]]
        self.max_len = int(meta.get("max_len", 64))

        adapter_cfg = self.model_dir / "adapter_config.json"
        if adapter_cfg.exists():
            cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
            self.base_model = meta.get("base_model") or cfg.get("base_model_name_or_path") or DEFAULT_BASE_MODEL
        else:
            self.base_model = meta.get("base_model") or str(self.model_dir)
        run = meta.get("run_name", "custom")
        best = meta.get("training", {}).get("best_epoch")
        self.model_version = f"{self.model_dir.name}:{run}" + (f"@epoch{best}" if best else "")

        if self.requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.requested_device

        has_tok = any((self.model_dir / f).exists() for f in ("tokenizer.json", "sentencepiece.bpe.model", "tokenizer_config.json"))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir if has_tok else self.base_model)

        if adapter_cfg.exists():
            from peft import PeftModel

            log.info("Loading base model %s (first run downloads ~2.5 GB from Hugging Face)…", self.base_model)
            base = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
            model = PeftModel.from_pretrained(base, str(self.model_dir)).merge_and_unload()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)

        if self.device == "cpu":
            model = model.float()   # fp16 weights are slow / unsupported on CPU
        model = model.to(self.device).eval()
        if self.device == "cpu" and self.quantize_cpu:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self._model = model

        for code in {c for d in self.directions for c in d}:
            if code not in self.code_map:
                raise ValueError(f"Language '{code}' has no NLLB tag in language_code_map")
        # warm-up so the first user request is not slowed down by lazy initialisation
        s, t = self.directions[0]
        self._generate(["Hello."], s, t)

    # -------------------------------------------------------------- translating
    @property
    def ready(self) -> bool:
        return self.state == EngineState.READY

    def supports(self, source_lang: str, target_lang: str) -> bool:
        return (source_lang, target_lang) in self.directions

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not self.ready:
            if self.state == EngineState.LOADING:
                raise ModelNotReady("The translation model is still loading. Please try again in a moment.")
            raise ModelNotReady(self.error or "The translation model is not loaded.")
        if not self.supports(source_lang, target_lang):
            raise UnsupportedDirection(f"Translation from {source_lang} to {target_lang} is not supported by the model.")

        lines = split_segments(text)
        if not any(normalize_source(seg, source_lang) for line in lines for seg in line):
            raise ValueError("Source text is empty after cleaning.")

        t0 = time.perf_counter()
        plan = [[c for c in (normalize_source(seg, source_lang) for seg in line) if c] for line in lines]
        segments = [seg for row in plan for seg in row]
        outputs, mean_logprobs, lengths = self._generate(segments, source_lang, target_lang)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        model_out = iter(outputs)
        rebuilt = []
        for row in plan:
            parts = [p for p in (next(model_out).strip() for _ in row) if p]
            if parts:
                rebuilt.append(" ".join(parts))
        translated = "\n".join(rebuilt)

        total = sum(lengths)
        confidence = None
        if total > 0 and all(math.isfinite(lp) for lp in mean_logprobs):
            confidence = round(math.exp(sum(lp * n for lp, n in zip(mean_logprobs, lengths)) / total), 4)
        return TranslationResult(translated, confidence, latency_ms)

    def _generate(self, segments: list[str], source_lang: str, target_lang: str):
        torch = self._torch
        tok, model = self._tokenizer, self._model
        with self._lock:   # tokenizer.src_lang is shared state
            tok.src_lang = self.code_map[source_lang]
            enc = tok(segments, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to(self.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                out = model.generate(
                    **enc,
                    forced_bos_token_id=tok.convert_tokens_to_ids(self.code_map[target_lang]),
                    num_beams=self.num_beams,
                    max_new_tokens=128,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            texts = tok.batch_decode(out.sequences, skip_special_tokens=True)

            gen = out.sequences[:, 1:]                       # drop decoder start token
            pad_id = tok.pad_token_id
            lengths = [int(n) for n in (gen != pad_id).sum(dim=1).tolist()]
            if getattr(out, "sequences_scores", None) is not None:
                # beam search: sum(log p) / length  -> mean token log-probability
                mean_lp = [float(v) for v in out.sequences_scores.float().tolist()]
            else:
                trans = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float()
                mask = (gen[:, : trans.shape[1]] != pad_id).float()
                mean_lp = ((trans * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).tolist()
        return texts, mean_lp, lengths

    def info(self) -> dict:
        return {
            "state": self.state.value,
            "error": self.error,
            "device": self.device,
            "base_model": self.base_model,
            "model_version": self.model_version,
            "num_beams": self.num_beams,
        }
