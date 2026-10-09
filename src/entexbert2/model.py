#!/usr/bin/env python3

"""
entexbert2.model — the 2-stage ASB model

  * task="regression"  (Stage-1 binding trunk):  mu = g(f(seq))
        One window -> one binding score,
        MSE on the (log1p fold-change) binding label (precision-weighted if depth column provided),
        Single-track (scalar) or multi-track (one score per tissue) via num_labels

  * task="classification"  (Stage-2 ASB head, symmetric contrast):
        s = ||P(h1) - P(h2)||, p = sigmoid(a*s + b),
        Predicts P(ASB) from the DISTANCE between the two haplotype representations in a LEARNED projection,
        Symmetric by construction,
        Loss = Precision-weighted BCE

File written by Amy Metrick in collaboration with Claude Science Opus 5 Agent
"""
from typing import Optional

import torch
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput

def get_activation_module(name: str) -> torch.nn.Module:
    name = name.lower()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "silu":
        return torch.nn.SiLU()
    raise ValueError(f"Unsupported head activation {name!r} (gelu|relu|tanh|silu).")

def build_prediction_head(
    input_size: int,
    output_size: int = 1,
    num_layers: int = 1,
    hidden_size: int = -1,
    activation: str = "gelu",
    dropout: float = 0.1,
) -> torch.nn.Module:
    """
    num_layers counts total Linear layers:
      1  -> Linear(input_size -> output_size)
      2  -> Linear(input->hidden) + activation + dropout + Linear(hidden->output)
      3+ -> deeper MLP
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
    if hidden_size == -1:
        hidden_size = input_size
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive or -1, got {hidden_size}.")
    if not (0 <= dropout < 1):
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

    # Linear head
    if num_layers == 1:
        return torch.nn.Linear(input_size, output_size)

    # MLP head
    layers = [torch.nn.Linear(input_size, hidden_size),
              get_activation_module(activation),
              torch.nn.Dropout(dropout)]

    for _ in range(num_layers - 2):
        layers += [torch.nn.Linear(hidden_size, hidden_size),
                   get_activation_module(activation),
                   torch.nn.Dropout(dropout)]
    layers.append(torch.nn.Linear(hidden_size, output_size))
    return torch.nn.Sequential(*layers)

class entexBERT2ForSequencePrediction(torch.nn.Module):
    """
    Stage-1 trunk (fine-tuned on binding affinity) -> Stage-2 head (task-selected)

    task="regression"  (Stage-1 binding trunk):
        forward scores one window through the trunk and regression head,
        returns the binding score mu = head(pool(seq))
        num_labels = 1 -> scalar (single-track);
        num_labels > 1 -> one score per tissue (multi-track, masked over observed tissues)

    task="classification" (Stage-2 ASB):
        forward projects each window's pooled representation with a SHARED LEARNED projection P (768 -> proj_dim),
        forms the distance s = ||P(h1) - P(h2)||_2, 
        returns logit ell = a*s + b -> P(ASB) = sigmoid(ell)
        Loss: precision-weighted BCE, L = sum_i w_i BCE(sigmoid(ell_i), y_i) / sum_i w_i,
        where w_i = n_eff(n_i) = n_i (1 + s) / (n_i + s), normalized to mean 1, n_i = privileged total read depth
    """

    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        pooling_mode: str = "center_mean",
        center_pool_width: int = 5,
        head_num_layers: int = 1,
        head_hidden_size: int = -1,
        head_activation: str = "gelu",
        head_dropout: float = 0.1,
        neff_s: float = 20.0,
        freeze_backbone: bool = False,
        task: str = "regression",
        num_labels: int = 1,
        proj_dim: int = 128, # classification projection dimension
    ):
        super().__init__()
        if pooling_mode not in ("cls", "center_mean", "mean"):
            raise ValueError(f"pooling_mode must be 'cls', 'center_mean', or 'mean', got {pooling_mode!r}.")
        if pooling_mode == "center_mean" and center_pool_width % 2 == 0:
            raise ValueError("center_pool_width must be odd for symmetric center pooling.")
        if neff_s <= 0:
            raise ValueError(f"neff_s must be > 0, got {neff_s}.")
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {task!r}.")
        num_labels = int(num_labels) # regression head width T (multi-track Stage-1 predicts one fold-change per tissue)
        if num_labels < 1:
            raise ValueError(f"num_labels must be >= 1, got {num_labels}.")
        if task == "classification" and num_labels != 1:
            raise ValueError(
                f"classification contrast head emits one P(ASB) logit; num_labels must be 1, "
                f"got {num_labels}."
            )
        self.pooling_mode = pooling_mode
        self.center_pool_width = int(center_pool_width)
        self.neff_s = float(neff_s)
        self.task = task
        self.num_labels = num_labels

        # Shared pretrained trunk (DNABERT-2-117M)
        self.backbone = transformers.AutoModel.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, trust_remote_code=True,
        )
        hidden_size = self.backbone.config.hidden_size
        dropout_prob = getattr(self.backbone.config, "hidden_dropout_prob", 0.1)
        self.dropout = torch.nn.Dropout(dropout_prob)

        head_input = hidden_size

        if task == "regression":
            self.main_head = build_prediction_head(
                input_size=head_input, output_size=self.num_labels, num_layers=head_num_layers,
                hidden_size=head_hidden_size, activation=head_activation, dropout=head_dropout,
            )
            # for classification only
            self.proj = None
            self.dist_a = None
            self.dist_b = None
        else: # classification
            if proj_dim <= 0:
                raise ValueError(f"proj_dim must be > 0 for classification, got {proj_dim}.")
            self.proj = build_prediction_head(
                input_size=head_input, output_size=proj_dim, num_layers=head_num_layers,
                hidden_size=head_hidden_size, activation=head_activation, dropout=head_dropout,
            )
            # logit = a * s + b
            # intialize so a = 1, b = 0 (identity mapping)
            # a computed via softplus(a_raw) at forward: log(1 + e^(a_raw)) to enforce monotonicity, smoothness
            # --> a_raw = ln(e − 1) = 0.5413, so that softplus(a_raw) = 1 at initialization
            self.dist_a = torch.nn.Parameter(torch.tensor(0.5413))
            self.dist_b = torch.nn.Parameter(torch.tensor(0.0))
            self.main_head = None

        if freeze_backbone:
            self.freeze_backbone()

    # ---- Stage-1 -> Stage-2 transfer ---------------------------------------
    def freeze_backbone(self):
        """Freeze the trunk so only the head trains (no collapse risk, reuses Stage 1)"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def init_backbone_from(self, checkpoint_path: str, map_location="cpu") -> int:
        """
        Load ONLY the `backbone.*` weights from a Stage-1 checkpoint state_dict into this fresh-head model;
        returns the number of backbone tensors loaded
        """
        sd = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(sd, dict) and "state_dict" in sd and "backbone.embeddings" not in str(sd.keys()):
            sd = sd["state_dict"]
        backbone_sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
        if not backbone_sd:
            raise ValueError(
                f"No 'backbone.*' keys found in {checkpoint_path}; got prefixes "
                f"{sorted({k.split('.')[0] for k in sd})}. Is this an entexBERT2 checkpoint?"
            )
        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)
        # 'missing' lists the fresh head params (these are expected); 'unexpected' should be empty!
        if unexpected:
            raise ValueError(f"Unexpected keys while loading backbone: {unexpected[:5]} ...")
        return len(backbone_sd)

    # ---- Pooling -----------------------------------------------------------
    def _pool(self, backbone_outputs, attention_mask=None):
        if hasattr(backbone_outputs, "last_hidden_state") and backbone_outputs.last_hidden_state is not None:
            seq = backbone_outputs.last_hidden_state
        elif isinstance(backbone_outputs, dict) and "last_hidden_state" in backbone_outputs:
            seq = backbone_outputs["last_hidden_state"]
        elif isinstance(backbone_outputs, (tuple, list)) and torch.is_tensor(backbone_outputs[0]) \
                and backbone_outputs[0].ndim == 3:
            seq = backbone_outputs[0]
        else:
            raise ValueError(f"Cannot extract token hidden states from {type(backbone_outputs)}.")

        if self.pooling_mode == "cls":
            return seq[:, 0, :]

        if self.pooling_mode == "mean":
            # masked all-token mean over the valid (non-pad) tokens of each window
            if attention_mask is None:
                return seq.mean(dim=1)
            m = attention_mask.unsqueeze(-1).to(seq.dtype)          # (B, L, 1)
            summed = (seq * m).sum(dim=1)                           # (B, H)
            denom = m.sum(dim=1).clamp(min=1.0)                     # (B, 1); avoid divide by 0 with clamp
            return summed / denom

        # center_mean: mean-pool center_pool_width tokens around the middle of each valid window
        bsz, max_len, _ = seq.shape
        half = self.center_pool_width // 2
        pooled = []
        for b in range(bsz):
            valid = int(attention_mask[b].sum().item()) if attention_mask is not None else max_len
            valid = max(valid, 1)
            center = valid // 2
            start = max(0, center - half)
            end = min(valid, center + half + 1)
            pooled.append(seq[b, start:end, :].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def _pool_one(self, input_ids, attention_mask, **kwargs):
        """One window -> pooled representation h in R^hidden"""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            return_dict=True, **kwargs)
        return self.dropout(self._pool(out, attention_mask=attention_mask))

    def _score_one(self, input_ids, attention_mask, **kwargs):
        """One window -> binding logit(s) + fused representation (regression head)

        Returns (B,) when num_labels = 1 and (B, T) when num_labels = T > 1
        """
        pooled = self._pool_one(input_ids, attention_mask, **kwargs)
        out = self.main_head(pooled)                      # (B, T)
        if self.num_labels == 1:
            out = out.squeeze(-1)                         # (B,)
        return out, pooled

    # ---- precision weight --------------------------------------------
    def _neff_weight(self, depth, like):
        """w_i = n_eff(n_i) normalized to mean 1; falls back to ones if depth is None"""
        if depth is None:
            return torch.ones_like(like)
        n = depth.float().view_as(like).clamp(min=1.0)
        s = self.neff_s
        w = n * (1.0 + s) / (n + s)          # n_eff, saturates at 1 + s
        return w / w.mean()                  # normalize to mean 1 (LR-invariant)

    # ---- Forward -----------------------------------------------------------
    def forward(
        self,
        input_ids=None,            # hap1 window (sequence1)
        attention_mask=None,
        input_ids_alt=None,        # hap2 window (sequence2)
        attention_mask_alt=None,
        labels=None,               # regression: binding label, (B,) scalar or (B,T) per-track; classification: 0/1 AS
        depth=None,                # n = read depth (privileged; weight only)
        label_mask=None,           # (B,T) 0/1 observed-tissue mask for multi-track Stage-1 (no mask => all observed)
        **kwargs,
    ):
        # classification: symmetric contrast (distance)
        if self.task == "classification":
            if input_ids_alt is None:
                raise RuntimeError(
                    "classification (contrast) head requires a paired (hap1, hap2) batch, but "
                    "input_ids_alt did not reach forward. Use --input_mode hap_pair and the twin collator."
                )
            h1 = self._pool_one(input_ids, attention_mask, **kwargs)
            h2 = self._pool_one(input_ids_alt, attention_mask_alt, **kwargs)
            z1 = self.proj(h1)
            z2 = self.proj(h2)
            s = torch.linalg.vector_norm(z1 - z2, dim=-1)      # >= 0, symmetric in (h1, h2)
            a = torch.nn.functional.softplus(self.dist_a)      # a > 0: distance monotone in P(ASB)
            ell = a * s + self.dist_b                          # logit of P(ASB)

            loss = None
            if labels is not None:
                y = labels.float().view_as(ell)
                w = self._neff_weight(depth, ell)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(ell, y, reduction="none")
                loss = (w * bce).mean()
            return SequenceClassifierOutput(loss=loss, logits=ell.unsqueeze(-1))

        # regression: Stage-1 binding trunk (single window)
        mu, _ = self._score_one(input_ids, attention_mask, **kwargs)
        # (B,) or (B,T) binding score

        # multi-track Stage-1 (num_labels > 1): masked multi-output MSE
        if self.num_labels > 1:
            loss = None
            if labels is not None:
                y = labels.float().view_as(mu)
                if label_mask is not None:
                    m = label_mask.float().view_as(mu)
                else:
                    m = torch.ones_like(mu)
                sq = m * (mu - y) ** 2
                denom = m.sum().clamp(min=1.0)
                loss = sq.sum() / denom
            return SequenceClassifierOutput(loss=loss, logits=mu)  # (B, T), no unsqueeze

        # scalar path (num_labels == 1): single-track Stage-1 binding score
        loss = None
        if labels is not None:
            y = labels.float().view_as(mu)
            w = self._neff_weight(depth, mu)
            loss = (w * (mu - y) ** 2).mean()

        return SequenceClassifierOutput(loss=loss, logits=mu.unsqueeze(-1))
