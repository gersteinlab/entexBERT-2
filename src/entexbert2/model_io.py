#!/usr/bin/env python3

"""
entexbert2.model_io - shared model I/O for entexBERT-2 evaluation, scoring, and interpretability

File written by Amy Metrick in collaboration with Claude Science Opus 4.8 Agent
"""

import glob
import json
import os
from typing import Optional

import torch
import transformers

from entexbert2.model import entexBERT2ForSequencePrediction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_run_config(checkpoint_dir: str) -> dict:
    """Read run_config.json written by the trainer (errors if missing)"""
    path = os.path.join(checkpoint_dir, "run_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No run_config.json in {checkpoint_dir}. It is written by the trainer at save "
            f"time; either re-run training with the current finetune script, or pass the "
            f"task/head fields explicitly."
        )
    with open(path) as f:
        return json.load(f)

def apply_overrides(run_config: dict, overrides: dict) -> dict:
    """Override run_config fields with any non-None values (from CLI)"""
    rc = dict(run_config)
    for k, v in (overrides or {}).items():
        if v is not None:
            rc[k] = v
    return rc

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def find_weights_file(checkpoint_dir: str) -> str:
    """
    Locate the saved weights; prefers the top-level save (best model, since the trainer
    runs with load_best_model_at_end), then falls back to the latest checkpoint-* dir
    """
    candidates = [
        os.path.join(checkpoint_dir, "pytorch_model.bin"),
        os.path.join(checkpoint_dir, "model.safetensors"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
    )
    for ckpt in reversed(ckpts):
        for name in ("pytorch_model.bin", "model.safetensors"):
            c = os.path.join(ckpt, name)
            if os.path.exists(c):
                return c

    raise FileNotFoundError(
        f"No pytorch_model.bin / model.safetensors found in {checkpoint_dir} "
        f"or its checkpoint-* subdirectories."
    )

def _load_state_dict(weights_path: str) -> dict:
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(weights_path)
    return torch.load(weights_path, map_location="cpu")

def load_model_weights(model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
    """Load a state dict into model, reporting missing/unexpected keys"""
    state_dict = _load_state_dict(weights_path)
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))

    print(f"Loaded weights from {weights_path}")
    print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if missing:
        print("  first missing:", missing[:10])
    if unexpected:
        print("  first unexpected:", unexpected[:10])

    # If the trained head didn't load, flag if the head params differ from expected by task
    task = getattr(model, "task", "regression")
    head_prefixes = ("main_head",) if task == "regression" else ("proj", "dist_a", "dist_b")
    head_missing = [k for k in missing if k.startswith(head_prefixes)]
    if head_missing:
        raise RuntimeError(
            f"{task} head weights did not load ({head_missing[:5]}...). The checkpoint and the "
            f"run_config architecture likely disagree. Refusing to run on an untrained head!"
        )
    return model

# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def build_model(run_config: dict, device: str = "cpu") -> torch.nn.Module:
    """Instantiate the trained architecture from a (possibly overridden) run_config"""
    if run_config.get("use_lora"):
        raise NotImplementedError(
            "run_config has use_lora=True; LoRA checkpoints need adapter handling that isn't "
            "wired up here yet. Train without LoRA or extend build_model."
        )

    head_num_layers = run_config["head_num_layers"]
    if head_num_layers >= 2 and "head_activation" not in run_config:
        raise KeyError(
            "run_config.json is missing 'head_activation', which this checkpoint needs: "
            f"head_num_layers={head_num_layers} builds an MLP head that requires a nonlinearity! "
            "Add the activation used at training time (gelu|relu|tanh|silu)."
        )

    model = entexBERT2ForSequencePrediction(
        model_name_or_path=run_config["model_name_or_path"],
        cache_dir=run_config.get("cache_dir"),
        pooling_mode=run_config["pooling_mode"],
        center_pool_width=run_config.get("center_pool_width", 5),
        head_num_layers=head_num_layers,
        head_hidden_size=run_config["head_hidden_size"],
        head_activation=run_config.get("head_activation", "gelu"),
        head_dropout=run_config.get("head_dropout", 0.1),
        neff_s=run_config.get("neff_s", 20.0),
        task=run_config.get("task", "regression"),
        proj_dim=run_config.get("proj_dim", 128),
        num_labels=run_config.get("num_labels", 1),
    )
    return model.to(device)

def load_tokenizer(run_config: dict):
    return transformers.AutoTokenizer.from_pretrained(
        run_config["model_name_or_path"],
        cache_dir=run_config.get("cache_dir"),
        model_max_length=run_config.get("model_max_length", 512),
        trust_remote_code=True,
    )

def load_model_and_tokenizer(checkpoint_dir: str, device: str = "cpu", overrides: dict = None):
    run_config = apply_overrides(load_run_config(checkpoint_dir), overrides or {})
    model = build_model(run_config, device=device)
    load_model_weights(model, find_weights_file(checkpoint_dir))
    model.eval()
    tokenizer = load_tokenizer(run_config)
    return model, tokenizer, run_config

# ---------------------------------------------------------------------------
# Inference helper (reuses the model's own backbone + pooling)
# ---------------------------------------------------------------------------

@torch.no_grad()
def logits_and_embeddings(model, input_ids, attention_mask,
                          input_ids_alt=None, attention_mask_alt=None,
                          return_pools=False):
    """
    Score sequence(s) with the model's backbone -> pooling -> head

    Returns:
        return_pools=False:  (logits, pooled_contrast)
        return_pools=True:   (logits, pooled_contrast, pool1, pool2)   (pool2=None if single sequence)
    """
    task = getattr(model, "task", "regression")

    # Route through model._pool_one so pooling matches training exactly
    def _pool(ids, mask):
        return model._pool_one(ids, mask)

    pool1 = _pool(input_ids, attention_mask)
    pool2 = None

    if task == "classification":
        if input_ids_alt is None:
            raise ValueError(
                "classification scoring requires paired inputs (hap1, hap2), but input was a single sequence"
            )
        pool2 = _pool(input_ids_alt, attention_mask_alt)
        z1 = model.proj(pool1)
        z2 = model.proj(pool2)
        pooled = z1 - z2 # projected contrast

        s = torch.linalg.vector_norm(z1 - z2, dim=-1, keepdim=True) # (N,1), >= 0
        a = torch.nn.functional.softplus(model.dist_a)
        logits = a * s + model.dist_b  # (N,1), logit P(ASB) = ell = a*||z1 - z2|| + b
        if return_pools:
            return logits, pooled, pool1, pool2
        return logits, pooled

    # regression (Stage-1 binding trunk): single-window score mu = head(pool1)
    logits = model.main_head(pool1)
    pooled = pool1

    if return_pools:
        return logits, pooled, pool1, pool2
    return logits, pooled


# ---------------------------------------------------------------------------
# Batched inference over a list of sequences (or [ref, alt] pairs)
# ---------------------------------------------------------------------------

def run_inference(checkpoint_dir, texts, batch_size=64, device="cpu",
                  overrides=None, dump_pools=False):
    """
    Load a checkpoint and score a list of inputs

    dump_pools=False (default):
        returns (logits, emb, run_config)
            logits : (N, 1)  regression: binding score mu ; classification: P(ASB) logit ell
            emb    : (N, D)  regression: pool1 (H) ; classification: proj contrast (proj_dim)
    dump_pools=True (pair inputs only):
        returns (logits, emb, pool_ref, pool_alt, run_config)
            pool_ref = pool(window1), pool_alt = pool(window2) -- RAW per-window pools

    Convention: window1 is the ref / hap1 window, window2 is the alt / hap2 window
    """
    model, tokenizer, run_config = load_model_and_tokenizer(
        checkpoint_dir, device=device, overrides=overrides)

    is_pair = len(texts) > 0 and isinstance(texts[0], (list, tuple))
    if dump_pools and not is_pair:
        raise ValueError("dump_pools=True requires paired inputs ([window1, window2]).")
    if getattr(model, "task", "regression") == "classification" and not is_pair:
        raise ValueError("classification scoring requires paired inputs ([window1, window2]).")
    mml = run_config.get("model_max_length", 512)

    all_logits, all_emb, all_ref, all_alt = [], [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if is_pair:
            ref = [b[0] for b in batch]
            alt = [b[1] for b in batch]
            enc_r = tokenizer(ref, return_tensors="pt", padding="longest",
                              max_length=mml, truncation=True)
            enc_a = tokenizer(alt, return_tensors="pt", padding="longest",
                              max_length=mml, truncation=True)
            out = logits_and_embeddings(
                model, enc_r["input_ids"].to(device), enc_r["attention_mask"].to(device),
                input_ids_alt=enc_a["input_ids"].to(device),
                attention_mask_alt=enc_a["attention_mask"].to(device),
                return_pools=dump_pools)
            if dump_pools:
                logits, pooled, pool1, pool2 = out
                all_ref.append(pool1.detach().cpu().numpy())
                all_alt.append(pool2.detach().cpu().numpy())
            else:
                logits, pooled = out
        else:
            enc = tokenizer(batch, return_tensors="pt", padding="longest",
                            max_length=mml, truncation=True)
            logits, pooled = logits_and_embeddings(
                model, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_logits.append(logits.detach().cpu().numpy())
        all_emb.append(pooled.detach().cpu().numpy())

    import numpy as np
    logits_out = np.concatenate(all_logits, axis=0)
    emb_out = np.concatenate(all_emb, axis=0)
    if dump_pools:
        return (logits_out, emb_out,
                np.concatenate(all_ref, axis=0), np.concatenate(all_alt, axis=0),
                run_config)
    return logits_out, emb_out, run_config
