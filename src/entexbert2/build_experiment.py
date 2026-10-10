#!/usr/bin/env python3

"""
entexbert2.build_experiment -- config-driven entexBERT-2 dataset runner (2-stage ASB pipeline)

Each experiment is a declarative config (YAML or JSON); this runner composes a row source
with a primary label and calls the source-agnostic build_dataset in entexbert2.build_inputs.

New formats are added by:
    1. implementing a RowSource / make_*_label_spec in build_inputs
    2. registering it in ROW_SOURCE_BUILDERS / LABEL_BUILDERS below. 

New *experiments* need no new code!

For 2-stage entexBERT-2 on ASB classification:

  Stage 1 (binding trunk):      row_source multi_tissue_peak + label bigwig / multitrack
  Stage 2 (ASB contrast head):  row_source hap_counts + label as_class, with
                                depth_col: n  (n carried through as the privileged weight)

Usage examples:
    python build_experiment.py configs/stage2_ctcf_asb.yaml
    python build_experiment.py exp.yaml --ref_fasta /data/hg38.fa --output_dir runs/run_name

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent

** Config formating details: **

===========================================================================
TOP LEVEL                                        (* = required)
===========================================================================
    ref_fasta *       path to the reference FASTA (CLI --ref_fasta overrides)
    output_dir *      where train/dev/test CSVs + .meta.csv sidecars are written
    row_source *      see ROW SOURCES
    primary_label *   see LABELS
    experiment        name recorded in finetune_settings.json          [experiment]
    window            see WINDOW                                       [{}]
    sequence          see SEQUENCE                                     [{}]
    split             see SPLIT                                        [{}]
    balance           see BALANCE                                      [{}]
    partition         see PARTITION (omit -> group-shuffle split)      [None]
    head              see HEAD (architecture only; task is derived)    [None]
    depth_col         source column carried as the `depth` weight      [None]
                      Stage 2 uses `n` (total reads -> n_eff weight)
    count_cols        extra columns copied into the Trainer CSVs       [None]
                      (multi-track Stage 1 uses this for y_track_*/m_track_*)

===========================================================================
ROW SOURCES   row_source: {type: ..., ...}
===========================================================================
  type: hap_counts          Stage 2. One row per (donor, locus) from build_hap_counts.py.
    path *                  CSV path; needs chr, ref_start, ref_allele,
                            hap1_allele, hap2_allele, k, n
    assay                   recorded on every row                      [None]
    donor                   recorded on every row                      [None]

  type: multi_tissue_peak   Stage 1. One row per consensus peak locus + sampled background.
    tissue_tracks *         [{tissue, peak_path, bigwig_path, [pval_path], [group]}, ...]
                            `group` = TF name; when EVERY entry has one, tracks are grouped
                            by TF instead of one track per tissue
    assay *                 recorded on every row
    donor *                 recorded on every row
    genome_sizes            chrom-sizes TSV; REQUIRED when background_ratio > 0 [None]
    format                  'narrowpeak' | 'bed3'                      [narrowpeak]
                            bed3 has no signalValue -> the summit tiebreak and the
                            mean_depth proxy become constant
    summit_mode             'summit' | 'midpoint'                      [summit]
    merge_window_bp         summits within this distance join one locus (single-linkage,
                            so dense regions can chain -- a diagnostic reports it)  [100]
    label_radius_bp         half-width of the signal read at each locus [32]
    background_ratio        background loci sampled per consensus locus [1.0]
    background_gap_bp       minimum distance from any peak             [1000]
    exclude_chroms          chromosomes dropped before everything      [None]
    seed                    background sampling seed                   [42]

===========================================================================
LABELS   primary_label: {type: ..., ...}        task_type drives the head
===========================================================================
  type: as_class            Stage 2 ASB target, binary 0/1     (classification)
    column                                                     [imbalance_significance]
    name                    output column name (alias: target_name) [as_class]

  type: bigwig              Stage 1 binding signal             (regression)
    path *                  BigWig file
    signal_mode             mean|max|min|sum|std|coverage|mean_nonzero|max_abs   [max]
    region                  'window' | 'snv' | 'snv_radius'    [snv_radius]
    radius_bp               used when region=snv_radius        [20]
    missing_value           returned when the region has no finite signal AND when
                            the row's chromosome is ABSENT from the BigWig. 0.0 is a
                            valid-looking label that the NaN filter will not drop;
                            pass .nan to have such rows dropped instead      [0.0]
    use_values              read per-base values and summarize here (safest, predictable
                            NaN handling) vs bw.stats zoom levels                [True]
    exact                   when use_values=False, pass exact= to bw.stats       [True]
    name / transform        see COMMON
    

  type: column              precomputed source column          (regression)
    column *                column to read
    name                    output column name                 [= column]
    transform               see COMMON

  type: multitrack          Stage 1, one target per track      (regression)
    num_tracks *            number of y_track_*/m_track_* columns
    anchor_column           scalar column kept alongside       [binding_label_raw]
    name                                                       [binding_label_raw]

  COMMON (all label types)
    name                    output column name
    transform               'identity' | 'log1p'
                            defaults: bigwig/multitrack -> log1p, column/as_class -> identity

===========================================================================
WINDOW   window: {...}
===========================================================================
    left_bp *               bases left of the anchor
    right_bp *              bases right of the anchor (window = left+right+1)
    offset_mode             'fixed' | 'uniform'                        [fixed]
    jitter_max_bp           max anchor jitter when offset_mode=uniform [0]
    chrom_sizes             chrom-sizes TSV. Without it windows cannot be bounds-checked
                            and an off-contig window aborts the build at the length check
                            instead of being dropped with a count              [None]

===========================================================================
SEQUENCE   sequence: {...}
===========================================================================
    input_mode              hap_pair | ref_single | ref_alt_pair |
                            ref_hap1_pair | ref_hap2_pair | hap1_single | hap2_single [hap_pair]
    mode                    'reference' (hg38 + allele swap) | 'personal'   [reference]
    hap_fastas              personal only: {ENC-00X: [hap1.fa, hap2.fa]}    [None]
    chains                  personal only: {ENC-00X: [maternal.chain, paternal.chain]} [None]
                            hap1 <- maternal, hap2 <- paternal; PersonalGenome re-derives
                            the assignment per locus by allele match

===========================================================================
SPLIT   split: {...}        (ignored when partition.enabled -- see PARTITION)
===========================================================================
    mode                    'train_dev_test' | 'train_only'            [train_dev_test]
    ratio                   (train, dev, test)                         [(0.8, 0.1, 0.1)]
    seed                    split + jitter seed                        [42]
    group                   'locus' -> group by locus_id (leakage prevention); '' disables
                            grouping AND the duplicate/label-conflict summary      [locus]
    skip_ambiguous          drop sequences containing N                [True]
    dedup_across_splits     drop a sequence already present in an earlier split    [True]
    exclude_loci_meta       .meta.csv whose locus_id values are removed before writing [None]

===========================================================================
BALANCE   balance: {...}
===========================================================================
    strategy                'none' | 'global_binary' | 'per_tissue_binary'    [none]
    label_col               column balanced on                   [imbalance_significance]
                            rows whose value is neither 0 nor 1 are excluded (reported)
    apply_to                'all'   -> balance BEFORE windowing/labelling/sequence
                                       extraction; later row drops can skew the realized
                                       ratio (reported when it drifts)
                            'train' -> balance the train split only, after every drop
                                       (use this when the ratio must hold)          [all]

===========================================================================
PARTITION   partition: {...}   deterministic, donor-invariant split; takes priority
                               over `split`. Omit (or enabled: false) to fall back.
===========================================================================
    enabled                 turn the partition on                      [False]
    fold_assignment         {chrom: fold index}; the chromosomes whose fold == fold_id
                            become TEST                                [{}]
    fold_id                 which fold is held out                     [0]
    bin_size                hashed genomic bin size for train/dev      [100000]
    salt                    hash salt. CHANGING IT REASSIGNS EVERY LOCUS and silently
                            breaks comparability with prior runs       [entexbert2_v1]
    train_frac_within_nontest   train:dev split of the non-test bins    [8/9]
    bin_test_frac           extra TEST fraction drawn from bins. With no fold_assignment
                            AND this at 0.0 there would be no test set (raises)     [0.0]
    bin_dev_frac            extra DEV fraction drawn from bins         [0.0]
    exclude_boundary        drop loci whose window straddles a bin edge (leakage
                            control; a no-op if boundary_bp <= 0, which now raises) [False]
    boundary_bp             half-width excluded at each bin edge
                            [max(left_bp, right_bp) + jitter_max_bp]

===========================================================================
HEAD   head: {...}          architecture only -- `task` is DERIVED from the label
===========================================================================
    task                    optional; if given it must MATCH the label's task_type,
                            else it raises. regression -> Stage-1 trunk;
                            classification -> Stage-2 contrast head
    head_num_layers         1 = linear head; >= 2 = MLP                [1]
    head_hidden_size        MLP width; -1 = same as hidden size        [-1]
    proj_dim                classification only: projection dim for the
                            contrast distance s = ||P(h1) - P(h2)||    [128]
    num_labels              DERIVED -- forced to 1, or to num_tracks for multitrack.
                            A conflicting value is overridden with a note.

    NOTE: any other key here is passed through to experiment_config.json for provenance
    but does NOT reach the model. The trainer flags set the architecture; this runner
    prints the mapping (see emit_finetune_settings). In particular head_activation is a
    finetune_entexbert2.py flag, and it is REQUIRED for head_num_layers >= 2 -- model_io
    raises when a checkpoint's run_config.json lacks it.
"""

import argparse
import dataclasses
import json
import os
import sys

from pyfaidx import Fasta

from entexbert2.build_inputs import (
    BalanceSpec,
    PartitionSpec,
    WindowSpec,
    MultiTissuePeakRowSource,
    HaplotypeCountRowSource,
    build_dataset,
    make_bigwig_label_spec,
    make_column_label_spec,
    log1p_transform,
    identity_transform,
)
from entexbert2.personal_seq import parse_chain_file, PersonalGenome

NONE_TISSUE_TOKENS = {None, "null", "NONE", "None", "none", "", "all", "ALL"}

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transform_fn(name):
    name = (name or "identity").lower()
    if name == "log1p":
        return log1p_transform
    if name == "identity":
        return identity_transform
    raise ValueError(f"Unsupported transform: {name!r}")

# ---------------------------------------------------------------------------
# Row source registry
# ---------------------------------------------------------------------------

def _narrowpeak_flag(fmt):
    """`format` selects narrowPeak vs BED3"""
    f = str(fmt).strip().lower()
    if f not in {"narrowpeak", "bed3"}:
        raise ValueError(f"row_source.format must be 'narrowpeak' or 'bed3', got {fmt!r}.")
    return f == "narrowpeak"

def _build_multi_tissue_peak_source(cfg):
    tracks = cfg["tissue_tracks"] # [{tissue, peak_path, bigwig_path}, ...]
    return MultiTissuePeakRowSource(
        datasets=tracks,
        assay=cfg["assay"],
        donor=cfg["donor"],
        genome_sizes_path=cfg.get("genome_sizes"),
        is_narrowpeak=_narrowpeak_flag(cfg.get("format", "narrowpeak")),
        summit_mode=cfg.get("summit_mode", "summit"),
        merge_window_bp=cfg.get("merge_window_bp", 100),
        label_radius_bp=cfg.get("label_radius_bp", 32),
        background_ratio=cfg.get("background_ratio", 1.0),
        background_gap_bp=cfg.get("background_gap_bp", 1000),
        exclude_chroms=cfg.get("exclude_chroms"),
        seed=cfg.get("seed", 42),
    )

def _build_hap_counts_source(cfg):
    return HaplotypeCountRowSource(
        counts_csv=cfg["path"],
        assay=cfg.get("assay"),
        donor=cfg.get("donor"),
    )

ROW_SOURCE_BUILDERS = {
    "multi_tissue_peak": _build_multi_tissue_peak_source,   # Stage 1 (binding trunk)
    "hap_counts": _build_hap_counts_source,                 # Stage 2 (ASB contrast head)
}

# ---------------------------------------------------------------------------
# Label registry
# ---------------------------------------------------------------------------

# Default post-fn transform per label type
_DEFAULT_TRANSFORM = {
    "bigwig": "log1p",
    "column": "identity",
    "as_class": "identity",     # binary AS label: read as-is (0/1), no transform
    "multitrack": "log1p",      # per-tissue fold-change, log1p like the mean binding target
}

def _label_name(cfg, fallback):
    return cfg.get("name") or cfg.get("target_name") or fallback

def _build_bigwig(cfg, tf):
    return make_bigwig_label_spec(
        name=_label_name(cfg, "bigwig"),
        bigwig_path=cfg["path"],
        mode=cfg.get("signal_mode", "max"),
        region=cfg.get("region", "snv_radius"),
        radius_bp=cfg.get("radius_bp", 20),
        transform_fn=tf,
        missing_value=cfg.get("missing_value", 0.0),
        use_values=bool(cfg.get("use_values", True)),
        exact=bool(cfg.get("exact", True)),
    )

def _build_column(cfg, tf):
    return make_column_label_spec(
        column=cfg["column"], name=cfg.get("name"), transform_fn=tf
    )

def _build_as_class(cfg, tf):
    # Binary ASB label (e.g. imbalance_significance, 0/1) read directly from a precomputed source column
    if tf is not identity_transform:
        raise ValueError(
            f"primary_label type 'as_class' is a binary 0/1 classification target; "
            f"transform={tf.__name__!r} would change the label values. Use transform: identity "
            f"(the default) or switch to type: column for a regression target.")

    return make_column_label_spec(
        column=cfg.get("column", "imbalance_significance"),
        name=cfg.get("name", "as_class"),
        task_type="classification",
        transform_fn=tf,
    )

def _build_multitrack(cfg, tf):
    # Multi-track Stage-1 binding target: predict one (transformed) value per tissue track
    n = int(cfg["num_tracks"])
    if n < 2:
        raise ValueError(f"multitrack label needs num_tracks >= 2, got {n}.")
    spec = make_column_label_spec(
        column=cfg.get("anchor_column", "binding_label_raw"),
        name=cfg.get("name", "binding_label_raw"),
        task_type="regression",
        transform_fn=tf,
    )
    # stash the track count + column names so run_from_config can wire count_cols + head width
    spec.multitrack_num_tracks = n
    spec.multitrack_y_cols = [f"y_track_{i}" for i in range(n)]
    spec.multitrack_m_cols = [f"m_track_{i}" for i in range(n)]
    return spec

LABEL_BUILDERS = {
    "bigwig": _build_bigwig,               # Stage 1 binding signal
    "column": _build_column,               # precomputed source column (regression)
    "as_class": _build_as_class,           # Stage 2 ASB target (classification, contrast head)
    "multitrack": _build_multitrack,       # Stage 1 multi-track binding (one target per tissue)
}

def build_label(cfg):
    ltype = cfg["type"]
    if ltype not in LABEL_BUILDERS:
        raise ValueError(f"Unknown label type {ltype!r}. Known: {sorted(LABEL_BUILDERS)}.")
    tf = get_transform_fn(cfg.get("transform", _DEFAULT_TRANSFORM.get(ltype, "identity")))
    return LABEL_BUILDERS[ltype](cfg, tf)

def build_source(cfg):
    stype = cfg["type"]
    if stype not in ROW_SOURCE_BUILDERS:
        raise ValueError(f"Unknown row_source type {stype!r}. Known: {sorted(ROW_SOURCE_BUILDERS)}.")
    return ROW_SOURCE_BUILDERS[stype](cfg)

# ---------------------------------------------------------------------------
# Config loading / helpers
# ---------------------------------------------------------------------------

def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path) as f:
        if ext in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as e:
                raise ImportError("PyYAML is required for YAML configs; use JSON or `pip install pyyaml`.") from e
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError(f"{path}: expected a YAML mapping of config sections, "
                                 f"got {type(cfg).__name__} (empty file?)")
            return cfg
        if ext == ".json":
            cfg = json.load(f)
            if not isinstance(cfg, dict):
                raise ValueError(f"{path}: expected a JSON mapping of config sections, "
                                 f"got {type(cfg).__name__} (empty file?)")
            return cfg
    raise ValueError(f"Unsupported config extension {ext!r}; use .yaml/.yml/.json.")

_SCHEMA = {
    None:            {"ref_fasta", "output_dir", "row_source", "primary_label", "experiment",
                      "window", "sequence", "split", "balance", "partition", "head",
                      "depth_col", "count_cols"},
    "window":        {"left_bp", "right_bp", "offset_mode", "jitter_max_bp", "chrom_sizes"},
    "sequence":      {"input_mode", "mode", "hap_fastas", "chains"},
    "split":         {"mode", "ratio", "seed", "group", "skip_ambiguous",
                      "dedup_across_splits", "exclude_loci_meta"},
    "balance":       {"strategy", "label_col", "apply_to"},
    "partition":     {"enabled", "fold_assignment", "fold_id", "bin_size", "salt",
                      "train_frac_within_nontest", "bin_test_frac", "bin_dev_frac",
                      "exclude_boundary", "boundary_bp"},
    "head":          {"task", "head_num_layers", "head_hidden_size", "proj_dim", "num_labels"},
}

def warn_unknown_keys(cfg):
    """Report config keys this runner never reads"""
    for section, known in _SCHEMA.items():
        block = cfg if section is None else cfg.get(section)
        if not isinstance(block, dict):
            continue
        unknown = sorted(set(block) - known)
        if unknown:
            where = "top level" if section is None else f"{section}:"
            print(f"[config] WARNING: unknown key(s) at {where} {unknown} -- ignored. "
                  f"Known: {sorted(known)}")

def load_exclude_loci(meta_paths):
    import pandas as pd
    loci = set()
    for p in meta_paths or []:
        df = pd.read_csv(p)
        if "locus_id" not in df.columns:
            raise ValueError(f"{p} has no 'locus_id' column (expected a *.meta.csv).")
        loci.update(df["locus_id"].astype(str).tolist())
    return loci

def build_partition_spec(cfg):
    """
    Build a PartitionSpec from an optional top-level `partition:` config block
    Returns None when the block is absent or partition.enabled is false 
    (in which case build_dataset falls back to the group-shuffle split)
    """
    pcfg = cfg.get("partition")
    if not pcfg or not pcfg.get("enabled", False):
        return None

    raw_assignment = pcfg.get("fold_assignment", {}) or {}
    fold_assignment = {str(k): int(v) for k, v in raw_assignment.items()}

    fold_id = int(pcfg.get("fold_id", 0))
    if fold_assignment and fold_id not in set(fold_assignment.values()):
        raise ValueError(
            f"partition.fold_id={fold_id} is not present in fold_assignment "
            f"(folds {sorted(set(fold_assignment.values()))}); the TEST set would be empty."
        )

    # boundary_bp defaults to the FURTHEST extent a window can reach from the anchor: max(left_bp, right_bp) + jitter_max_bp
    _wcfg = cfg.get("window", {}) or {}
    _default_bp = (max(int(_wcfg.get("left_bp", 0)), int(_wcfg.get("right_bp", 0)))
                   + int(_wcfg.get("jitter_max_bp", 0) or 0))
    return PartitionSpec(
        enabled=True,
        bin_size=int(pcfg.get("bin_size", 100_000)),
        salt=str(pcfg.get("salt", "entexbert2_v1")),
        fold_assignment=fold_assignment,
        fold_id=fold_id,
        train_frac_within_nontest=float(pcfg.get("train_frac_within_nontest", 8.0 / 9.0)),
        bin_test_frac=float(pcfg.get("bin_test_frac", 0.0)),
        bin_dev_frac=float(pcfg.get("bin_dev_frac", 0.0)),
        exclude_boundary=bool(pcfg.get("exclude_boundary", False)),
        boundary_bp=int(pcfg.get("boundary_bp", _default_bp)),
    )

def resolve_head(head, primary_label):
    """
    Derive/validate the finetune head config from the primary label's task_type
    """
    head = dict(head or {})
    task = primary_label.task_type  # comes from the label

    if "task" in head and head["task"] != task:
        raise ValueError(
            f"head.task={head['task']!r} disagrees with the primary label's task_type={task!r}. "
            f"Omit head.task (it is derived) or fix it."
        )
    head["task"] = task

    if task not in ("regression", "classification"):
        raise ValueError(
            f"Supported tasks: 'regression' | 'classification'; got {task!r}."
        )
    # Head width
    n_tracks = getattr(primary_label, "multitrack_num_tracks", None)
    if task == "classification":
        if head.get("num_labels", 1) != 1:
            print(f"  note: classification forces num_labels=1 (config had {head.get('num_labels')}).")
        head["num_labels"] = 1
    elif n_tracks is not None:
        # multi-track Stage-1 binding: T outputs, one per tissue
        head["num_labels"] = int(n_tracks)
        head["multitrack"] = True
    else:
        if head.get("num_labels", 1) != 1:
            print(f"  note: single-track regression forces num_labels=1 (config had {head.get('num_labels')}).")
        head["num_labels"] = 1

    head.setdefault("head_num_layers", 1)
    head.setdefault("head_hidden_size", -1)
    if task == "classification":
        # projection dim d for the shared P: hidden -> d used to form the contrast distance
        head.setdefault("proj_dim", 128)
    return head

def emit_finetune_settings(head, output_dir, primary_name, depth_col):
    """Print the finetune settings recorded for this dataset (verified flags only)"""
    print("\nFinetune settings (map to finetune_entexbert2.py):")
    print(f"  --task {head['task']}")
    print(f"  --head_num_layers {head['head_num_layers']}  (1 = linear, >1 = MLP)")
    if head["head_num_layers"] >= 2:
        print(f"  --head_activation gelu|relu|tanh|silu   (REQUIRED for an MLP head; "
              f"model_io raises if run_config.json lacks it)")
    if head.get("num_labels", 1) > 1:
        print(f"  --num_labels {head['num_labels']}   (REQUIRED for the multi-track trunk: the "
              f"trainer defaults to 1, which would build a scalar head against "
              f"{head['num_labels']}-track targets and pin the scalar label_names path)")
    if head["head_hidden_size"] != -1:
        print(f"  --head_hidden_size {head['head_hidden_size']}")
    if head["task"] == "classification":
        print(f"  --proj_dim {head.get('proj_dim', 128)}   (shared projection dim d for the contrast distance)")
        print(f"  --balanced_sampler True   (class-balanced batches)")
    if depth_col:
        print(f"  --neff_s <s>   (privileged precision weight from '{depth_col}' -> depth)")
    print(f"  trainer data path: {output_dir}")
    print(f"  primary label column: {primary_name}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run an entexBERT-2 dataset generation experiment from a config")
    p.add_argument("config", help="Path to the experiment config (.yaml/.yml/.json)")
    p.add_argument("--ref_fasta", default=None, help="Override ref_fasta from the config")
    p.add_argument("--output_dir", default=None, help="Override output_dir from the config")
    return p.parse_args()

def build_personal_genomes(seqcfg):
    """Construct {donor: PersonalGenome} from a personal `sequence:` config section

    Per donor:
        hap_fastas: {ENC-00X: [hap1.fa, hap2.fa]} 
        chains:     {ENC-00X: [maternal.chain, paternal.chain]}

    For entex personal genomes v2:
    hap1 -> maternal chain
    hap2 -> paternal chain
    PersonalGenome re-derives the hap1<->hap2 assignment for sex-chromosomes per locus by allele-match (to handle edge cases)
    """
    hap_fastas = seqcfg.get("hap_fastas") or {}
    chains = seqcfg.get("chains") or {}
    donors = sorted(set(hap_fastas) & set(chains))
    _unpaired = sorted(set(hap_fastas) ^ set(chains))
    if _unpaired:
        raise ValueError(
            f"sequence.hap_fastas and sequence.chains disagree on donors {_unpaired}: each donor "
            f"needs BOTH entries. hap_fastas has {sorted(hap_fastas)}, chains has {sorted(chains)}")
    if not donors:
        raise ValueError("sequence.mode='personal' needs matching donors in "
                         "sequence.hap_fastas and sequence.chains.")
    genomes = {}
    for d in donors:
        for _key, _val in (("hap_fastas", hap_fastas[d]), ("chains", chains[d])):
            if not isinstance(_val, (list, tuple)) or len(_val) != 2:
                raise ValueError(f"sequence.{_key}[{d!r}] must be a 2-element list "
                                 f"[hap1/maternal, hap2/paternal], got {_val!r}.")

        h1_fa, h2_fa = hap_fastas[d]
        mat_chain, pat_chain = chains[d]
        genomes[d] = PersonalGenome(
            parse_chain_file(mat_chain), Fasta(h1_fa),
            parse_chain_file(pat_chain), Fasta(h2_fa),
            donor=d,
        )
        print(f"  personal[{d}]: hap1<-{os.path.basename(h1_fa)}/{os.path.basename(mat_chain)}  "
              f"hap2<-{os.path.basename(h2_fa)}/{os.path.basename(pat_chain)}")
    return genomes

def run_from_config(cfg, ref_fasta=None, output_dir=None):
    """
    Build a dataset from a config dict,
    ref_fasta / output_dir override the config when provided
    
    returns the PRE-WRITE DataFrame from build_dataset (before the writer's N-sequence,
    exclude_loci, cross-split-dedup and boundary filtering), so len(df) can exceed the
    rows actually written; read train/dev/test.csv for the final data
    
    Importable for notebooks/tests!
    """
    name = cfg.get("experiment", "experiment")
    warn_unknown_keys(cfg)
    ref_fasta_path = ref_fasta or cfg["ref_fasta"]
    output_dir = output_dir or cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    source = build_source(cfg["row_source"])
    primary_label = build_label(cfg["primary_label"])

    # window config
    wcfg = cfg.get("window", {})
    window_spec = WindowSpec(
        left_bp=wcfg["left_bp"],
        right_bp=wcfg["right_bp"],
        chrom_sizes_path=wcfg.get("chrom_sizes"),
        offset_mode=wcfg.get("offset_mode", "fixed"),
        jitter_max_bp=wcfg.get("jitter_max_bp", 0),
    )

    # split config
    scfg = cfg.get("split", {})
    seed = scfg.get("seed", 42)
    split_mode = scfg.get("mode", "train_dev_test")
    
    _group = scfg.get("group", "locus")
    if _group not in {"locus", "", "none", None}:
        raise ValueError(
            f"split.group must be 'locus' (group rows by locus_id -- the leakage control that "
            f"keeps a locus out of two splits) or ''/'none' to disable it; got {_group!r}.")
    group_cols = ["locus_id"] if _group == "locus" else []

    split_ratio = tuple(scfg.get("ratio", (0.8, 0.1, 0.1)))
    skip_ambiguous = scfg.get("skip_ambiguous", True)
    exclude_loci = load_exclude_loci(scfg.get("exclude_loci_meta")) or None
    dedup_across_splits = scfg.get("dedup_across_splits", True)

    # balance config
    bcfg = cfg.get("balance", {})
    balance_spec = BalanceSpec(
        strategy=bcfg.get("strategy", "none"),
        label_col=bcfg.get("label_col", "imbalance_significance"),
        random_state=seed,
    )
    balance_split = bcfg.get("apply_to", "all")  # "all" = balance before split; "train" = balance train split only
    if balance_split not in {"all", "train"}:
        raise ValueError(f"balance.apply_to must be 'all' or 'train', got {balance_split!r}.")

    input_mode = cfg.get("sequence", {}).get("input_mode", "hap_pair")
    sequence_mode = cfg.get("sequence", {}).get("mode", "reference")   # "reference" (hg38+swap) | "personal"
    depth_col = cfg.get("depth_col")   # Stage 2: privileged precision weight (w = n_eff)
    count_cols = list(cfg.get("count_cols") or [])  # optional: extra columns carried into train.csv

    # Multi-track Stage-1: carry the per-tissue target + mask columns into train.csv via the count_cols passthrough
    _mt_y = getattr(primary_label, "multitrack_y_cols", None)
    if _mt_y:
        _mt_m = getattr(primary_label, "multitrack_m_cols", [])
        for c in list(_mt_y) + list(_mt_m):
            if c not in count_cols:
                count_cols.append(c)
    count_cols = count_cols or None

    # Optional hybrid cross-individual partition (held-out test chrom(s) + hashed genomic bins) for leak-free cross-donor eval
    # None -> fall back to the group-shuffle split
    partition_spec = build_partition_spec(cfg)

    # Head is derived from the label's task_type
    head = resolve_head(cfg.get("head"), primary_label)

    print(f"Experiment: {name}")
    print(f"  source:     {source.source_type} (has_variants={source.has_variants})")
    print(f"  primary:    {primary_label.name} [{primary_label.task_type}]")
    print(f"  input_mode: {input_mode}")
    print(f"  window:     L{window_spec.left_bp}/R{window_spec.right_bp} "
          f"offset={window_spec.offset_mode} jitter={window_spec.jitter_max_bp}")
    print(f"  split:      {split_mode} group={'locus' if group_cols else 'none'} "
          f"exclude={len(exclude_loci) if exclude_loci else 0} loci")
    if depth_col:
        print(f"  depth_col:  {depth_col} (-> 'depth' privileged weight)")

    if partition_spec is not None:
        _test_chroms = sorted(c for c, f in partition_spec.fold_assignment.items()
                              if f == partition_spec.fold_id)
        print(f"  partition:  hybrid bin_size={partition_spec.bin_size} "
              f"fold_id={partition_spec.fold_id} test_chroms={_test_chroms} "
              f"(overrides group-shuffle)")
    print(f"  output_dir: {output_dir}")

    # Save all the experiments's actual setup parameters, dataset split info, for reproducibility
    _mt_y = getattr(primary_label, "multitrack_y_cols", None)
    if _mt_y:
        _tissues = list(getattr(source, "tissues", []))
        if len(_tissues) != len(_mt_y):
            raise ValueError(
                f"multitrack num_tracks={len(_mt_y)} disagrees with the source's tissue count "
                f"{len(_tissues)} ({_tissues}). Set primary_label.num_tracks to the number of "
                f"tissue tracks in row_source.tissue_tracks."
            )
        with open(os.path.join(output_dir, "tracks.json"), "w") as f:
            json.dump({"num_tracks": len(_tissues),
                       "tracks": [{"index": i, "tissue": t, "y_col": f"y_track_{i}",
                                   "m_col": f"m_track_{i}"} for i, t in enumerate(_tissues)]},
                      f, indent=2)
        print(f"  multitrack:  {len(_tissues)} tissue tracks -> tracks.json")

    partition_resolved = dataclasses.asdict(partition_spec) if partition_spec is not None else None
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump({"experiment": name, "resolved": cfg, "head_resolved": head,
            "partition_resolved": partition_resolved,
            # the raw `cfg` above records only what the user WROTE; 
            # this records what the runner actually used, so defaults are reproducible without the code version
            "runtime_resolved": {
                "window": dataclasses.asdict(window_spec),
                "balance": dataclasses.asdict(balance_spec),
                "balance_split": balance_split,
                "split_mode": split_mode, "split_ratio": list(split_ratio),
                "seed": seed, "group_cols": group_cols,
                "skip_ambiguous": skip_ambiguous,
                "dedup_across_splits": dedup_across_splits,
                "input_mode": input_mode, "sequence_mode": sequence_mode,
                "depth_col": depth_col, "count_cols": count_cols,
            },
            "ref_fasta": ref_fasta_path, "output_dir": output_dir}, f, indent=2)

    print("Loading reference FASTA...")
    ref = Fasta(ref_fasta_path)

    personal_genomes = None
    if sequence_mode == "personal":
        print("Building personal genomes (chain lift + haplotype FASTAs)...")
        personal_genomes = build_personal_genomes(cfg.get("sequence", {}))

    df = build_dataset(
        row_source=source,
        output_dir=output_dir,
        ref_fasta=ref,
        primary_label=primary_label,
        window_spec=window_spec,
        input_mode=input_mode,
        sequence_mode=sequence_mode,
        personal_genomes=personal_genomes,
        balance_spec=balance_spec,
        balance_split=balance_split,
        split_ratio=split_ratio,
        seed=seed,
        skip_ambiguous=skip_ambiguous,
        group_cols=group_cols,
        split_mode=split_mode,
        exclude_loci=exclude_loci,
        dedup_sequences_across_splits=dedup_across_splits,
        partition_spec=partition_spec,
        depth_col=depth_col,
        count_cols=count_cols,
    )

    print(f"\nDone building! Final rows: {len(df)}")
    emit_finetune_settings(head, output_dir, primary_label.name, depth_col)
    return df

def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("experiment", os.path.splitext(os.path.basename(args.config))[0])
    run_from_config(cfg, ref_fasta=args.ref_fasta, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
