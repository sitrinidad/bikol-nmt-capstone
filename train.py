"""Fine-tuning script (LoRA) for Bikol NMT (Pandan <-> Virac <-> English).

Base model : facebook/nllb-200-distilled-600M
Method     : one LoRA adapter shared by six directions
Selection  : best epoch by VALIDATION BLEU (not loss); the test set is never loaded here.

Usage (Colab T4 GPU):
    pip install -q transformers peft accelerate sacrebleu sentencepiece pandas matplotlib
    python train.py

Input  : data/directional/{train,val}.<src>-<tgt>.jsonl   (made by the Phase 1 notebook)
         or, if that folder is missing, data/train.csv and data/val.csv
         (columns: eng, bcl_South, bcl_North)
Output : models/lora_adapter/   adapter, tokenizer, bikol_meta.json
         results/loss_history.csv and results/figures/loss_curves.png
"""
import argparse, dataclasses, inspect, json, math, os, time

import numpy as np
import pandas as pd
import sacrebleu
import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)

MODEL_NAME = "facebook/nllb-200-distilled-600M"
# NLLB-200 has no Bikol tag, so each language borrows its OWN existing tag.
# This is a technical device: the adapter redefines what the tag means.
CODE = {"eng": "eng_Latn", "bcl_North": "tgl_Latn", "bcl_South": "war_Latn"}
DIRECTIONS = [("bcl_North", "eng"), ("eng", "bcl_North"), ("bcl_South", "eng"),
              ("eng", "bcl_South"), ("bcl_South", "bcl_North"), ("bcl_North", "bcl_South")]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_torchao_ok(min_version="0.16.0"):
    """Colab ships an old torchao (0.10) that makes peft raise ImportError. LoRA does not need torchao,
    so we remove it instead of asking the user to do it by hand."""
    import importlib, importlib.metadata as md, subprocess, sys
    from packaging import version
    try:
        v = md.version("torchao")
    except md.PackageNotFoundError:
        return
    if version.parse(v) < version.parse(min_version):
        print(f"torchao {v} is incompatible with this peft version -> uninstalling it (not needed for LoRA)")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False, capture_output=True)
        importlib.invalidate_caches()
        for k in [k for k in sys.modules if k == "torchao" or k.startswith("torchao.")]:
            del sys.modules[k]
        try:
            import peft.import_utils as iu
            iu.is_torchao_available.cache_clear()
        except Exception:
            pass


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--adapter_dir", default="models/lora_adapter")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--checkpoint_dir", default="checkpoints")  # keep out of git
    p.add_argument("--seed", type=int, default=42)             # same seed as the split
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)            # keeps alpha/r = 2
    p.add_argument("--lora_dropout", type=float, default=0.20) # raised after v3 memorised the data
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.15)
    p.add_argument("--src_token_dropout", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)          # epochs without a validation-BLEU gain
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--warmup_frac", type=float, default=0.06)
    p.add_argument("--max_len", type=int, default=64)
    return p.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_split(data_dir, split):
    """Return {direction: [{'src_lang','tgt_lang','src','tgt'}, ...]} for train or val."""
    out = {}
    d = os.path.join(data_dir, "directional")
    if os.path.isdir(d):
        for s, t in DIRECTIONS:
            out[(s, t)] = read_jsonl(os.path.join(d, f"{split}.{s}-{t}.jsonl"))
        return out
    df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"), dtype=str)
    for s, t in DIRECTIONS:
        out[(s, t)] = [{"src_lang": s, "tgt_lang": t, "src": a, "tgt": b} for a, b in zip(df[s], df[t])]
    return out


def main():
    a = get_args()
    set_seed(a.seed)
    os.makedirs(a.adapter_dir, exist_ok=True)
    os.makedirs(f"{a.results_dir}/figures", exist_ok=True)

    train_by_dir, val_by_dir = load_split(a.data_dir, "train"), load_split(a.data_dir, "val")
    train_rows = [r for rows in train_by_dir.values() for r in rows]
    val_rows = [r for rows in val_by_dir.values() for r in rows]
    print(f"train examples: {len(train_rows)} | val examples: {len(val_rows)} (test set NOT loaded)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(rows):
        feats = []
        for r in rows:  # the language tags are set per example, so one model serves all directions
            tokenizer.src_lang, tokenizer.tgt_lang = CODE[r["src_lang"]], CODE[r["tgt_lang"]]
            enc = tokenizer(r["src"], text_target=r["tgt"], max_length=a.max_len, truncation=True)
            feats.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": enc["labels"]})
        return feats

    train_data, val_data = encode(train_rows), encode(val_rows)

    @torch.no_grad()
    def translate(model, texts, src_lang, tgt_lang, num_beams=1, bs=64):
        model.eval()
        tokenizer.src_lang = CODE[src_lang]
        forced = tokenizer.convert_tokens_to_ids(CODE[tgt_lang])
        order = np.argsort([-len(t) for t in texts])
        out = [None] * len(texts)
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = tokenizer([texts[j] for j in idx], return_tensors="pt", padding=True,
                            truncation=True, max_length=a.max_len).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                gen = model.generate(**enc, forced_bos_token_id=forced, num_beams=num_beams, max_new_tokens=96)
            for j, d in zip(idx, tokenizer.batch_decode(gen, skip_special_tokens=True)):
                out[j] = d
        return out

    class BleuSelect(TrainerCallback):
        """After each epoch decode the validation set (greedy), keep the best-BLEU adapter, stop early.
        Validation LOSS starts rising early on this small corpus while BLEU still improves,
        so loss is the wrong selector (see Figure 1 in the report)."""
        def __init__(self):
            self.best, self.best_epoch, self.best_state, self.bad, self.history = -1.0, 0, None, 0, []

        def on_epoch_end(self, args, state, control, model=None, **kw):
            ep, bl, cf = int(round(state.epoch)), [], []
            for (s, t), rows in val_by_dir.items():
                hyps = translate(model, [r["src"] for r in rows], s, t)
                refs = [r["tgt"] for r in rows]
                bl.append(sacrebleu.corpus_bleu(hyps, [refs]).score)
                cf.append(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score)
            b, c = float(np.mean(bl)), float(np.mean(cf))
            self.history.append({"epoch": ep, "val_BLEU": b, "val_chrF++": c})
            if b > self.best + 1e-6:
                self.best, self.best_epoch, self.bad = b, ep, 0
                self.best_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            else:
                self.bad += 1
            print(f"[epoch {ep}] val BLEU {b:.2f} | chrF++ {c:.2f} | best {self.best:.2f} @ epoch {self.best_epoch}")
            if self.bad >= a.patience:
                print(f"early stop: no validation-BLEU gain for {a.patience} epochs")
                control.should_training_stop = True
            return control

    class LSTrainer(Trainer):
        """Label smoothing and source-token dropout applied only while training.
        Done here (not via TrainingArguments) so `labels` still reach NLLB, which shifts them itself."""
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            if model.training and a.src_token_dropout > 0:  # noise on the INPUT only
                ids = inputs["input_ids"]
                drop = ((torch.rand(ids.shape, device=ids.device) < a.src_token_dropout)
                        & (inputs["attention_mask"] == 1) & (ids != tokenizer.eos_token_id))
                drop[:, 0] = False  # never drop the language tag
                inputs = {**inputs, "input_ids": ids.masked_fill(drop, tokenizer.unk_token_id)}
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100,
                label_smoothing=a.label_smoothing if model.training else 0.0)
            return (loss, outputs) if return_outputs else loss

    # base model stays frozen (fp32); only the LoRA weights train, fp16 comes from AMP
    ensure_torchao_ok()  # must run before peft builds the adapter
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.lora_dropout, target_modules=TARGET_MODULES,
                     bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad), "a non-LoRA weight is trainable!"

    steps_per_epoch = math.ceil(len(train_data) / a.batch_size)
    kw = dict(output_dir=a.checkpoint_dir, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
              per_device_eval_batch_size=a.batch_size * 2, learning_rate=a.lr, lr_scheduler_type="linear",
              warmup_steps=int(a.warmup_frac * steps_per_epoch * a.epochs), optim="adamw_torch",
              weight_decay=a.weight_decay, fp16=(DEVICE == "cuda"), logging_strategy="epoch",
              save_strategy="no", report_to="none", seed=a.seed, remove_unused_columns=False,
              label_names=["labels"])
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    kw["eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"] = "epoch"  # renamed in newer versions
    targs = TrainingArguments(**kw)
    tkw = ({"processing_class": tokenizer} if "processing_class" in inspect.signature(Trainer.__init__).parameters
           else {"tokenizer": tokenizer})
    cb = BleuSelect()
    trainer = LSTrainer(model=model, args=targs, train_dataset=train_data, eval_dataset=val_data,
                        data_collator=DataCollatorForSeq2Seq(tokenizer, model=None, padding=True,
                                                             pad_to_multiple_of=8, label_pad_token_id=-100),
                        callbacks=[cb], **tkw)
    t0 = time.time()
    trainer.train()
    print(f"training finished in {(time.time() - t0) / 60:.1f} min")

    set_peft_model_state_dict(model, cb.best_state)  # restore the best-BLEU epoch, not the last one
    print(f"restored best-BLEU weights from epoch {cb.best_epoch} (mean val BLEU {cb.best:.2f})")

    # loss history (used for Figure 1)
    log = trainer.state.log_history
    tr = {int(round(h["epoch"])): h["loss"] for h in log if "loss" in h and "eval_loss" not in h}
    ev = {int(round(h["epoch"])): h["eval_loss"] for h in log if "eval_loss" in h}
    hist = pd.DataFrame({"epoch": sorted(ev)})
    hist["train_loss"], hist["val_loss"] = hist.epoch.map(tr), hist.epoch.map(ev)
    hist["val_BLEU"] = hist.epoch.map({h["epoch"]: h["val_BLEU"] for h in cb.history})
    hist["val_chrF++"] = hist.epoch.map({h["epoch"]: h["val_chrF++"] for h in cb.history})
    hist.to_csv(f"{a.results_dir}/loss_history.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(hist.epoch, hist.train_loss, "o-", label="training loss")
    ax.plot(hist.epoch, hist.val_loss, "s-", label="validation loss")
    ax.axvline(cb.best_epoch, ls="--", c="gray", label=f"best-BLEU epoch = {cb.best_epoch}")
    ax.set(xlabel="epoch", ylabel="cross-entropy loss", title="Training and validation loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{a.results_dir}/figures/loss_curves.png", dpi=150)

    # export the adapter + tokenizer + metadata that the API reads
    model.save_pretrained(a.adapter_dir)
    tokenizer.save_pretrained(a.adapter_dir)
    best_val_loss = float(hist.loc[hist.epoch == cb.best_epoch, "val_loss"].iloc[0])
    meta = {"base_model": MODEL_NAME, "directions": DIRECTIONS, "language_code_map": CODE, "max_len": a.max_len,
            "lora": {"r": a.r, "alpha": a.alpha, "dropout": a.lora_dropout, "target_modules": TARGET_MODULES},
            "training": {"lr": a.lr, "epochs": a.epochs, "batch_size": a.batch_size, "seed": a.seed,
                         "best_epoch": cb.best_epoch, "best_val_BLEU": cb.best, "best_val_loss": best_val_loss,
                         "label_smoothing": a.label_smoothing, "src_token_dropout": a.src_token_dropout,
                         "weight_decay": a.weight_decay}}
    with open(f"{a.adapter_dir}/bikol_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"adapter saved to {a.adapter_dir} | trainable parameters: {n_train:,}")


if __name__ == "__main__":
    main()
