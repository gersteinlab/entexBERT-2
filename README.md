# entexBERT-2
### **Two-stage** transfer learning with a **DNA transformer** for predicting **allele-specific activity**

built on a modified DNABERT-2 backbone and trained on EN-TEx allele-specific read counts.

The model is a **twin** (Siamese) network: two allelic windows share one backbone and one projection, and the allele-specific call is a logistic function of the distance between their projected representations.

---

## The two-stage design

1. **Stage 1 — binding trunk (regression):** the pre-trained DNABERT-2 backbone is fine-tuned to predict continuous
binding signal (`log1p` of fold-change over control) from a single reference window
    - This teaches the trunk *where* a factor binds
    - Input mode: `ref_single`
    - Task: `regression`

> Note that multi-track supervision (one output per tissue) is also supported via `primary_label.num_tracks` but not used in the accompanying paper's results

2. **Stage 2 — allele-specific head (classification):** the Stage-1 trunk is **frozen** and a small distance-logistic head is trained on top
    - For a heterozygous SNV, the two haplotype windows are each embedded to a projection `z`, and
        ```
        s = ||z1 - z2||          contrast distance
        p = sigma(a * s + b)       a = softplus(a_raw) >= 0
        ```

        so a larger allelic representational distance gives a higher allele-specific probability.

    - Input mode: `hap_pair`
    - Task: `classification`
    - Read counts enter into the loss function **only at training time**, as a privileged precision weight (`n_eff`) — the deployed model needs only sequence

---

## Install

```bash
conda create -n eb2 python=3.8 && conda activate eb2

pip install -e .                 # core: dataset building only (no torch)
pip install -e ".[model]"        # + torch, transformers, peft  -> training and scoring
pip install -e ".[analysis]"     # + matplotlib, logomaker, memelite -> figures
pip install -e ".[all]"          # everything
```

`import entexbert2` works in an environment without torch and only pulls a dependency when you touch a name that needs it:

```python
from entexbert2 import build_dataset              # needs pyBigWig, not torch
from entexbert2 import load_model_and_tokenizer   # needs torch
```

---

## Quickstart

The dataset build is **config-driven**: `build_experiment.py` reads a YAML config and writes
`train/dev/test.csv` plus `.meta.csv` sidecars, `experiment_config.json`, and (for multi-track
runs) `tracks.json`. `finetune_entexbert2.py` then trains from that directory.

```bash
# 1. Stage 1 -- build + train the binding trunk (regression)
python -m entexbert2.build_experiment configs/stage1_binding.yaml \
    --ref_fasta "$REF" --output_dir "$BUILD_S1"

python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path DNABERT-2-117M-attention --data_path "$BUILD_S1" \
    --task regression --input_mode ref_single \
    --pooling_mode center_mean --center_pool_width 5 \
    --head_num_layers 1 --model_max_length 512 \
    --metric_for_best_model spearman \
    --num_train_epochs 3 --learning_rate 1e-3 --bf16 \
    --output_dir "$TRUNK"

# 2. Stage 2 -- build + train the AS head on the frozen trunk (classification)
python -m entexbert2.build_experiment configs/stage2_asb.yaml \
    --ref_fasta "$REF" --output_dir "$BUILD_S2"

python -m entexbert2.finetune_entexbert2 \
    --model_name_or_path DNABERT-2-117M-attention --data_path "$BUILD_S2" \
    --task classification --input_mode hap_pair \
    --pooling_mode center_mean --center_pool_width 5 \
    --head_num_layers 2 --head_hidden_size 128 --head_activation gelu --proj_dim 128 \
    --balanced_sampler True --neff_s 20 \
    --init_backbone_from "$TRUNK/pytorch_model.bin" --freeze_backbone True \
    --metric_for_best_model auroc \
    --num_train_epochs 3 --learning_rate 1e-3 --bf16 \
    --output_dir "$HEAD"
```

Note that `build_experiment` prints the trainer flags implied by the dataset it just built (task, head
width, whether `--head_activation` and `--num_labels` are required

A console script is installed as a shorthand for the build step:

```bash
entexbert2-build configs/stage1_binding.yaml --ref_fasta "$REF" --output_dir "$BUILD_S1"
```

---

## Model arguments (`finetune_entexbert2.py`)

| Argument | Values | Meaning |
|---|---|---|
| `--task` | `regression` \| `classification` | Stage-1 binding trunk vs Stage-2 allele-specific head. Selects head topology and loss. |
| `--input_mode` | `ref_single` \| `hap_pair` | One reference window (Stage 1) vs the (hap1, hap2) twin (Stage 2). |
| `--pooling_mode` | `center_mean` \| `cls` \| `mean` | How token states become one sequence representation. `center_mean` averages the central tokens. |
| `--center_pool_width` | int (default 5) | Tokens averaged by `center_mean`. Must be **odd**. |
| `--num_labels` | int (default 1) | Head width. **Required for multi-track Stage 1** — set it to the number of tissue tracks, or the trainer builds a scalar head against multi-track targets. |
| `--head_num_layers` | int (default 1) | 1 = linear head; >= 2 = MLP with `--head_hidden_size`. |
| `--head_hidden_size` | int (default -1) | Hidden width when `head_num_layers >= 2` (-1 = backbone width). |
| `--head_activation` | `gelu` \| `relu` \| `tanh` \| `silu` (default `gelu`) | MLP activation. Recorded in `run_config.json`; `model_io` requires it when `head_num_layers >= 2`. |
| `--head_dropout` | float (default 0.1) | Head dropout at train time (inert at inference). |
| `--proj_dim` | int (default 128) | Width of the shared projection used to form the contrast distance (classification only). |
| `--balanced_sampler` | bool (default False) | Class-balanced batches (Stage 2). Do **not** also subsample the data. |
| `--neff_s` | float (default 20.0) | Saturation cap `s` for the privileged depth weight `w = n(1+s)/(n+s)`. Loss-only — unused at inference. |
| `--init_backbone_from` | path | Initialize the backbone from a checkpoint (Stage 2 loads the Stage-1 trunk here). |
| `--freeze_backbone` | bool (default False) | Freeze the backbone (True for the frozen-trunk Stage-2 arm). |
| `--use_lora` | bool (default False) | Parameter-efficient fine-tuning; see `--lora_r`, `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`. |

Everything else is standard HuggingFace `TrainingArguments` (`--learning_rate`,
`--num_train_epochs`, `--per_device_train_batch_size`, `--bf16`, `--model_max_length`, ...).
Note `--metric_for_best_model` defaults to `spearman`, which is ideal for Stage 1; pass `auroc` for
Stage 2.

Training writes `run_config.json` alongside the weights. `model_io` reads it back to reconstruct the exact architecture

---

## Config schema

Every config key, its default, and both runnable examples are documented in the
`build_experiment.py` module docstring:

```bash
python -c "import entexbert2.build_experiment as m; print(m.__doc__)"
```

The main blocks are `row_source` (where windows and labels come from), `primary_label`
(its `task_type` derives the head), `window`, `sequence`, `split`,
`partition`, `balance`, and `depth_col`. Unrecognised keys are reported at build time.

### Leakage controls

`partition` implements the hybrid split the cross-individual evaluation depends on: whole
held-out chromosomes for test, plus hashed genomic bins for train/dev, with optional
`exclude_boundary` so no window straddles a bin edge. The assignment depends only on
`(chromosome, position)`, so a locus lands in the same split for every donor and assay.
`split.group: locus` additionally keeps all rows for one SNV together.

### Personal genomes

With `sequence.mode: personal`, haplotype windows are extracted from donor-specific genome
FASTAs by lifting each reference locus through maternal and paternal chain files, rather than
substituting alleles into the reference. Haplotype assignment is re-derived per locus by
allele match, so phasing conventions do not have to be assumed. Requires `sequence.hap_fastas`
and `sequence.chains` to name the same donors.

---

## Repository layout

```
src/entexbert2/
├── __init__.py              public API (import entexbert2 pulls no third-party package)
├── model.py                 2-stage model
├── model_io.py              run_config/weights loading, run_inference
├── build_inputs.py          row sources, labels, windowing, splits, balancing, writing
├── build_experiment.py      config-driven dataset builder (python -m entexbert2.build_experiment)
├── eval_utils.py            shared scoring helpers (window building, hetSNV loading)
└── personal_seq.py          chain-file lift and personal-genome window extraction
DNABERT-2-117M-attention/    vendored backbone modeling code (see Third-party code)
```

Training, scoring and analysis modules are being migrated into this package and will appear
under `src/entexbert2/` and `src/entexbert2/analysis/`.

---

## Third-party code

`DNABERT-2-117M-attention/` is a vendored copy of
[DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M), distributed under the
**Apache License 2.0** — see `DNABERT-2-117M-attention/LICENSE`.

**Modified:** `bert_layers.py` is patched so that `output_attentions=True` returns the
content + padding attention with the ALiBi positional bias removed. This is a reporting-only
change — the model's own forward pass still uses content + ALiBi + padding — so the weights
are identical to the upstream release. The directory is vendored rather than fetched from the
Hub so that the backbone the checkpoints were trained against stays pinned.

---

## License

MIT — see [LICENSE](LICENSE).
