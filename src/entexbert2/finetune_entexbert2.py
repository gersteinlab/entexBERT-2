"""
This script is a modified version of the DNABERT-2 finetuning script,
found at https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/train.py

This modified version supports:
- Continuous label prediction via a linear regression head
- 2-stage fine-tune

All modifications to the original script are wrapped in comments in the following format:
### NEW/MODIFIED: [description of addition/modification] ####
...
#############################################################

Last modified: 10/13/2026 by Amy Metrick
"""

import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, Union

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

## NEW IMPORTS ##################################################
from scipy.stats import pearsonr, spearmanr # regression metrics
from entexbert2.model import entexBERT2ForSequencePrediction
#################################################################


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M", metadata={"help": "path to the DNABERT-2 backbone"}) # MODIFIED: change from facebook/opt-125m to DNABERT-2
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    # NEW: entexBERT-2 head / pooling architecture (used by entexBERT2ForSequencePrediction) ################
    pooling_mode: str = field(default="center_mean", metadata={"help": "'center_mean', 'mean', or 'cls'"})
    center_pool_width: int = field(default=5, metadata={"help": "odd width of the center-mean pool window"})
    head_num_layers: int = field(default=1, metadata={"help": "1 = linear head, >=2 = MLP head g_phi"})
    head_hidden_size: int = field(default=-1, metadata={"help": "MLP hidden size (-1 = auto)"})
    head_activation: str = field(default="gelu")
    head_dropout: float = field(default=0.1)
    num_labels: int = field(default=1, metadata={"help": "regression head width T (multi-track Stage-1 = #tissue tracks; 1 = scalar)"})
    # NEW: privileged precision weighting w =  n(1+s)/(n+s) #################################################
    neff_s: float = field(default=20.0, metadata={"help": "n_eff saturation cap s (0 = unweighted)"})
    # NEW: classification (contrast head) projection dimension d for delta = ||P(h1) - P(h2)|| ##############
    proj_dim: int = field(default=128, metadata={"help": "classification: shared projection dim d"})
    # NEW: 2-stage transfer learning ########################################################################
    init_backbone_from: Optional[str] = field(default=None,
        metadata={"help": "path to a Stage-1 checkpoint; loads backbone.* weights only"})
    freeze_backbone: bool = field(default=False,
        metadata={"help": "freeze the backbone (Stage 2: train the twin head only)"})
    #########################################################################################################

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    # MODIFIED: removed kmer argument (was DNABERT(1) legacy)
    # NEW: regression task capabilities and streamline haplotype pair input format ######################################################################################
    task: str = field(default="regression", metadata={"help": "'regression' (Stage-1 binding trunk, single window) or 'classification' (Stage-2 ASB contrast head)"})
    input_mode: str = field(default="hap_pair", metadata={"help": "'hap_pair' (twin, Stage 2) or 'ref_single' (one window, Stage 1)"})
    #####################################################################################################################################################################

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    # NEW: regression selects the best checkpoint by Spearman, greater is better!
    metric_for_best_model: str = field(default="spearman")
    greater_is_better: bool = field(default=True)
    #############################################################################
    # NEW: class-balanced batch sampling (via WeightedRandomSampler) for the rare-positive classification tasks
    balanced_sampler: bool = field(default=False, metadata={"help": "class-balanced batches (classification)"})
    ###########################################################################################################

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

# MODIFIED: removed unused "get_alter_of_dna_sequence" function from original DNABERT-2 train code
# MODIFIED: removed "generate_kmer)str" and "load_or_generate_kmer" since entexBERT-2 uses only BPE tokenization

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    # MODIFIED: removed kmer argument (DNABERT(1) legacy)
    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 input_mode: str = "hap_pair"): # NEW: hap_pair reads two windows per example

        super(SupervisedDataset, self).__init__()

        # MODIFIED: load data from the disk ##############################################################
             # read by column name: sequence1,sequence2,label[,depth])
             # NOTE: original DNABERT-2 code inferred single-vs-paired sequence input from the column count

        with open(data_path, "r") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"{data_path} is empty.")
        cols = rows[0].keys()

        if input_mode == "hap_pair":
            if "sequence1" not in cols or "sequence2" not in cols:
                raise ValueError(f"hap_pair needs sequence1,sequence2 columns; got {list(cols)}.")
            logging.warning(f"hap_pair: two windows per example (task={input_mode and 'see --task'})...")
            texts = [[r["sequence1"], r["sequence2"]] for r in rows]
        else:
            seqcol = "sequence" if "sequence" in cols else "sequence1"
            logging.warning("ref_single: single window per example (task={input_mode and 'see --task'})...")
            texts = [r[seqcol] for r in rows]

        # NEW: multi-track Stage-1 binding
        y_track_cols = sorted([c for c in cols if c.startswith("y_track_")],
                              key=lambda c: int(c.split("_")[-1]))
        self.multitrack = len(y_track_cols) > 0
        if self.multitrack:
            m_track_cols = [f"m_track_{c.split('_')[-1]}" for c in y_track_cols]
            missing_m = [c for c in m_track_cols if c not in cols]
            if missing_m:
                raise ValueError(f"multi-track CSV missing mask columns {missing_m}; "
                                 f"have y_track cols {y_track_cols}.")
            self.num_tracks = len(y_track_cols)
            # labels: (N, T) per-tissue targets; label_mask: (N, T) "observed" flags
            labels = [[float(r[c]) for c in y_track_cols] for r in rows]
            self.label_mask = [[float(r[c]) for c in m_track_cols] for r in rows]
        else:
            # NEW: float labels for regression tasks (scalar path)
            labels = [float(r["label"]) for r in rows]
            self.label_mask = None

        # NEW: optional privileged depth column n (the precision weight)
        self.depth = [float(r["depth"]) for r in rows] if "depth" in cols else None
        ####################################################################################################

        # MODIFIED: removed kmer != -1 branch

        # MODIFIED: for hap_pair, tokenize the two windows SEPARATELY rather than [SEP]-concatenating
        self.input_mode = input_mode
        if input_mode == "hap_pair":
            enc1 = tokenizer([t[0] for t in texts], return_tensors="pt", padding="longest",
                             max_length=tokenizer.model_max_length, truncation=True)
            enc2 = tokenizer([t[1] for t in texts], return_tensors="pt", padding="longest",
                             max_length=tokenizer.model_max_length, truncation=True)
            self.input_ids = enc1["input_ids"]
            self.attention_mask = enc1["attention_mask"]
            self.input_ids_alt = enc2["input_ids"]
            self.attention_mask_alt = enc2["attention_mask"]
        else:
            enc = tokenizer(texts, return_tensors="pt", padding="longest",
                            max_length=tokenizer.model_max_length, truncation=True)
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.input_ids_alt = None
            self.attention_mask_alt = None

        self.labels = labels
        # NEW: multi-track -> T outputs; scalar path -> 1
        self.num_labels = self.num_tracks if self.multitrack else 1
        ############################################################################################

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # MODIFIED: emit the twin pair + float label (+ depth weight when present)
        item = dict(input_ids=self.input_ids[i], labels=self.labels[i])
        if self.input_ids_alt is not None:
            item["input_ids_alt"] = self.input_ids_alt[i]
        if self.depth is not None:
            item["depth"] = self.depth[i]
        # NEW: multi-track observed-tissue mask
        if self.label_mask is not None:
            item["label_mask"] = self.label_mask[i]
        return item
        ###########################################################################

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        # MODIFIED: float labels (regression target) instead of .long() class ids
        batch = dict(
            input_ids=input_ids,
            labels=torch.tensor(labels, dtype=torch.float),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        # NEW: pad + attach the second (alt) window when present (twin input)
        if "input_ids_alt" in instances[0]:
            alt = torch.nn.utils.rnn.pad_sequence(
                [instance["input_ids_alt"] for instance in instances],
                batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch["input_ids_alt"] = alt
            batch["attention_mask_alt"] = alt.ne(self.tokenizer.pad_token_id)
        
        # NEW: carry the privileged depth weight through to the loss (crucially NOT the model input)
        if "depth" in instances[0]:
            batch["depth"] = torch.tensor([instance["depth"] for instance in instances],
                                          dtype=torch.float)
        
        # NEW: multi-track observed-tissue mask -> (B, T), forwarded to the masked MSE loss
        if "label_mask" in instances[0]:
            batch["label_mask"] = torch.tensor(
                [instance["label_mask"] for instance in instances], dtype=torch.float)
        
        return batch

# NEW: regression metrics ######################################################################
def calculate_regression_metrics(predictions: np.ndarray, labels: np.ndarray):
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    out = {
        "mse": float(np.mean((pred - y) ** 2)),
        "pearson": float(pearsonr(pred, y)[0]) if len(y) > 2 else float("nan"),
        "spearman": float(spearmanr(pred, y).correlation) if len(y) > 2 else float("nan"),
    }
    direction = (y > 0).astype(int) # peak-vs-background sign for the single-sequence regression trunk
    if 0 < direction.sum() < len(direction):
        out["auroc"] = float(sklearn.metrics.roc_auc_score(direction, pred))
    else:
        out["auroc"] = float("nan")
    return out
##############################################################################################

# NEW: multi-track regression metrics (Stage-1, one target per tissue) ########################
# predictions/y/mask are (N, T), only observed entries (mask=1) count
#   mse       : masked-pooled MSE over all observed (locus, tissue) entries
#   pearson   : masked-pooled Pearson over the same
#   spearman  : MEAN per-track Spearman (each tissue's own rank correlation, 
#               averaged over tissues with >=2 observed rows)
#   auroc     : masked-pooled peak-vs-low direction (y>median) AUROC (for context only)
def calculate_multitrack_metrics(predictions, y, mask):
    pred = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.asarray(mask, dtype=float)
    if pred.ndim == 1:  # in case single-track slipped through
        pred = pred.reshape(-1, 1); y = y.reshape(-1, 1); m = m.reshape(-1, 1)
    obs = m > 0.5
    pv, yv = pred[obs], y[obs]
    out = {
        "mse": float(np.mean((pv - yv) ** 2)) if pv.size else float("nan"),
        "n_obs": float(obs.sum()),
        "n_tracks": float(pred.shape[1]),
    }
    out["pearson"] = float(pearsonr(pv, yv)[0]) if pv.size > 2 else float("nan")
    # mean per-track Spearman over tissues with enough observed rows
    rhos = []
    for t in range(pred.shape[1]):
        col = obs[:, t]
        if int(col.sum()) > 2:
            r = spearmanr(pred[col, t], y[col, t]).correlation
            if np.isfinite(r):
                rhos.append(r)
    out["spearman"] = float(np.mean(rhos)) if rhos else float("nan")
    # pooled direction AUROC vs the observed-target median (peak-vs-low proxy)
    if yv.size > 2:
        thr = np.median(yv)
        direction = (yv > thr).astype(int)
        if 0 < direction.sum() < len(direction):
            out["auroc"] = float(sklearn.metrics.roc_auc_score(direction, pv))
        else:
            out["auroc"] = float("nan")
    else:
        out["auroc"] = float("nan")
    return out
##############################################################################################

# NEW: classification metrics (contrast head) ##################################################################
# The model emits ell = logit P(ASB) (higher = more allele-specific)
#   PRIMARY (threshold-free, honest for rare-positive data): auroc, auprc;
#           rank on ell directly (NOT |ell|)
#   DIAGNOSTIC for context only (threshold at ell=0, i.e. p=0.5): mcc/precision/recall/f1
# NOTE: this function logically replaces original calculate_metric_with_sklearn function from DNABERT-2 train.py
def calculate_classification_metrics(predictions: np.ndarray, labels: np.ndarray):
    ell = np.asarray(predictions, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    yb = (y > 0.5).astype(int)
    n = len(yb); n_pos = int(yb.sum())
    out = {
        "bce": float(np.mean(np.logaddexp(0.0, ell) - yb * ell)), # BCE-with-logits, unweighted
        "n_pos": float(n_pos),
        "pos_rate": float(n_pos / n) if n else float("nan"),
    }
    if 0 < n_pos < n:
        out["auroc"] = float(sklearn.metrics.roc_auc_score(yb, ell))         # PRIMARY
        out["auprc"] = float(sklearn.metrics.average_precision_score(yb, ell))  # PRIMARY
        pred = (ell > 0.0).astype(int)                                       # threshold at p=0.5
        out["mcc"] = float(sklearn.metrics.matthews_corrcoef(yb, pred))      # DIAGNOSTIC
        out["precision"] = float(sklearn.metrics.precision_score(yb, pred, zero_division=0))
        out["recall"] = float(sklearn.metrics.recall_score(yb, pred, zero_division=0))
        out["f1"] = float(sklearn.metrics.f1_score(yb, pred, zero_division=0))
    else:
        for key in ("auroc", "auprc", "mcc", "precision", "recall", "f1"):
            out[key] = float("nan")
    return out
##############################################################################################################

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
# def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
#     if isinstance(logits, tuple):  # Unpack logits if it's a tuple
#         logits = logits[0]

#     if logits.ndim == 3:
#         # Reshape logits to 2D if needed
#         logits = logits.reshape(-1, logits.shape[-1])

#     return torch.argmax(logits, dim=-1)

# MODIFIED: for regression, logits ARE the prediction ######################################
def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 2 and logits.shape[-1] > 1:
        return logits # multi-track: (N, T)
    return logits.reshape(-1) # scalar: (N, 1) -> (N,)
############################################################################################

"""
Compute metrics used for huggingface trainer.
""" 
# MODIFIED: task-aware metric function builder #############################
def make_compute_metrics(task):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task == "regression" and isinstance(labels, (tuple, list)):
            y, mask = labels[0], labels[1]
            return calculate_multitrack_metrics(predictions, y, mask)
        if task == "classification":
            return calculate_classification_metrics(predictions, labels)
        return calculate_regression_metrics(predictions, labels)
    return compute_metrics

# NEW: Trainer that can draw class-balanced training batches for the rare-positive AS task ###
# When training_args.balanced_sampler is set, each example is sampled with probability
# inversely proportional to its class frequency (so batches are ~50/50 pos/neg on expectation)
class BalancedTrainer(transformers.Trainer):
    def _make_balanced_sampler(self):
        labels = np.asarray(self.train_dataset.labels, dtype=float)
        yb = (labels > 0.5).astype(int)
        n_pos, n_neg = int(yb.sum()), int((1 - yb).sum())
        if n_pos == 0 or n_neg == 0:
            return None  # degenerate; fall back to default shuffling
        # per-class weight = 1 / class_count -> equal expected mass on pos and neg
        w = np.where(yb == 1, 1.0 / n_pos, 1.0 / n_neg).astype(np.float64)
        return torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(w, dtype=torch.double),
            num_samples=len(w), replacement=True,
        )

    def get_train_dataloader(self):
        if not getattr(self.args, "balanced_sampler", False):
            return super().get_train_dataloader()
        sampler = self._make_balanced_sampler()
        if sampler is None:
            return super().get_train_dataloader()
        logging.warning("Using class-balanced WeightedRandomSampler for training batches.")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
################################################################################################

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # NEW: the twin/depth columns are non-standard Trainer inputs -- make sure we keep them!
    training_args.remove_unused_columns = False
    ########################################################################################

    # NEW: metric_for_best_model defaults to 'spearman', change to AUROC if task = classification
    if data_args.task == "classification" and training_args.metric_for_best_model == "spearman":
        training_args.metric_for_best_model = "auroc"
        logging.warning("task=classification: 'spearman' is not emitted for this task; "
                        "using metric_for_best_model='auroc', "
                        "change config and re-run to use a different classification metric).")
    ###############################################################################################

    # NEW: validate input_mode ###########################################################################
    if data_args.input_mode not in ("hap_pair", "ref_single"):
        raise ValueError(f"--input_mode must be 'hap_pair' or 'ref_single', got {data_args.input_mode!r}.")
    #######################################################################################################

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator [MODIFIED: added input_mode=data_args.input_mode for hap_pair]
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                    data_path=os.path.join(data_args.data_path, "train.csv"),
                                    input_mode=data_args.input_mode)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                    data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                    input_mode=data_args.input_mode)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                    data_path=os.path.join(data_args.data_path, "test.csv"), 
                                    input_mode=data_args.input_mode)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # NEW: reconcile head width with the dataset before building the model #################
    _ds_labels = train_dataset.num_labels
    if model_args.num_labels == 1 and _ds_labels > 1:
        raise ValueError(
            f"{data_args.data_path}/train.csv has {_ds_labels} y_track_* columns (multi-track) "
            f"but --num_labels is 1, which would train a scalar head. Pass --num_labels {_ds_labels}.")
    if model_args.num_labels != _ds_labels:
        raise ValueError(
            f"--num_labels {model_args.num_labels} disagrees with the dataset's {_ds_labels} "
            f"target column(s) -- fix them to match!")
    #########################################################################################

    # NEW: multi-track Stage-1 needs the observed tissue mask to reach compute_metrics ######
    if model_args.num_labels > 1:
        training_args.label_names = ["labels", "label_mask"]
    else:
        # no label mask for other input paradigms
        training_args.label_names = ["labels"]
    #########################################################################################

    # MODIFIED: load entexBERT-2 model
    model = entexBERT2ForSequencePrediction(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        pooling_mode=model_args.pooling_mode,
        center_pool_width=model_args.center_pool_width,
        head_num_layers=model_args.head_num_layers,
        head_hidden_size=model_args.head_hidden_size,
        head_activation=model_args.head_activation,
        head_dropout=model_args.head_dropout,
        neff_s=model_args.neff_s,
        task=data_args.task, # selects head topology
        num_labels=model_args.num_labels, # regression head width T (multi-track Stage-1)
        proj_dim=model_args.proj_dim, # classification projection dimension
    )

    # NEW: 2-stage transfer ################################################################################
        # Load a Stage-1 binding trunk (backbone only) -> optionally freeze it for Stage-2 (twin head train)
    if model_args.init_backbone_from:
        model.init_backbone_from(model_args.init_backbone_from)
    if model_args.freeze_backbone:
        model.freeze_backbone()
    ########################################################################################################

    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # NEW: persist the architecture so model_io.py can rebuild the model exactly for scoring ###
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "run_config.json"), "w") as f:
        json.dump({
            "model_name_or_path": model_args.model_name_or_path,
            "cache_dir": training_args.cache_dir,
            "model_max_length": training_args.model_max_length,
            "pooling_mode": model_args.pooling_mode,
            "center_pool_width": model_args.center_pool_width,
            "head_num_layers": model_args.head_num_layers,
            "head_hidden_size": model_args.head_hidden_size,
            "head_activation": model_args.head_activation,
            "head_dropout": model_args.head_dropout,
            "neff_s": model_args.neff_s,
            "task": data_args.task,
            "num_labels": model_args.num_labels, # regression head width T (multi-track Stage-1)
            "proj_dim": model_args.proj_dim,
            "input_mode": data_args.input_mode,
            "use_lora": model_args.use_lora,
            "init_backbone_from": model_args.init_backbone_from,
            "freeze_backbone": model_args.freeze_backbone,
            "balanced_sampler": training_args.balanced_sampler,
            "seed": training_args.seed,
        }, f, indent=2)
    ##############################################################################################

    # define trainer [MODIFIED: BalancedTrainer adds optional class-balanced sampling]
    trainer = BalancedTrainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=make_compute_metrics(data_args.task),
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
                                   
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    train()