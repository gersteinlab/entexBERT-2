"""
entexbert2.build_inputs

Use: dataset construction for 2-stage SFT pipeline built on a DNABERT-2 backbone

Driven by two classes:
1. RowSource (sequence)
2. LabelSpec (label)

Separates resulting full (sequence, label) datasets into leak-free train/dev/test CSVs

Current scope: TF binding affinity, ASB datasets

New sources/labels can be added by:
1. Writing the desired RowSource construction and make_*_label_spec function here
2. Add the new RowSource class to run_experiment.ROW_SOURCE_BUILDERS
3. Add the new LabelSpec builder function to run_experiment.LABEL_BUILDERS

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""

import os
import hashlib
import pandas as pd
import numpy as np
import pyBigWig

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Any

def log1p_transform(x: float) -> float:
    """
    log1p transform for nonnegative signal
    """
    x = float(x)

    if not np.isfinite(x):
        return 0.0

    if x < 0:
        x = 0.0

    return float(np.log1p(x))

def identity_transform(x):
    return x

@dataclass
class LabelSpec:
    """
    Specification for one label column

    - fn: takes in a row-like object (sequence, label) and returns the raw label value
    - transform_fn: applied after fn to encode a smoother prediction target (e.g. taking the log)
    - required_columns: columns the fn needs to be present on each row (validated at in build_dataset() against the row source's actual columns)
    """
    name: str
    fn: Callable[[pd.Series], float]
    task_type: str = "regression" # default regression
    transform_fn: Callable[[float], float] = identity_transform
    required_columns: List[str] = field(default_factory=list)

@dataclass
class WindowSpec:
    """
    Anchor-centered window specification

    - left_bp: number of nucleotides to the left of the anchor (e.g. SNV or binding peak) to include in the window
    - right_bp: number of nucleotides to the right of the anchor to include in the window
    - offset_mode: "fixed" (anchor always at offset left_bp) 
                    or "uniform" (anchor offset jittered uniformly within +/- jitter_max_bp of left_bp, per example);
                    the realized per-example offset is recorded downstream as anchor_offset_seq1[/seq2]
    - jitter_max_bp: max +/- jitter of the anchor offset within the window
        * Must be <= min(left_bp, right_bp) so the anchor always stays inside the fixed-length window *

    Note: there is always a fixed window length = left_bp + right_bp + 1 (anchor nucleotide) 
    """
    left_bp: int
    right_bp: int
    chrom_sizes_path: Optional[str] = None # path to .chrom.sizes file to check windows don't overflow chromosome boundary
    offset_mode: str = "fixed"
    jitter_max_bp: int = 0

@dataclass
class BalanceSpec:
    """
    Balancing/subsampling strategy:
        - "none": do not balance
        - "global_binary": balance by label_col across full dataset
        - "per_tissue_binary": balance by label_col separately within each tissue
    """
    strategy: str = "none"
    label_col: str = "imbalance_significance"
    random_state: int = 42

@dataclass
class PartitionSpec:
    """
    Hybrid cross-individual split
    (only active when enabled=True; otherwise, build_dataset falls back to the plain group-shuffle split):
        - test chroms present -> hold out whole chromosome(s) for TEST, hash the rest into TRAIN/DEV by genomic bin
        - no test chroms + bin_test_frac > 0 -> pure genomic-bin 3-way split (TEST/DEV/TRAIN all assigned by the bin hash)
        - bin_size / salt / ratios travel in the resolved config
        - K-fold-ready: fold_assignment maps every chromosome to a fold index and fold_id selects which fold is the current test set
    """
    enabled: bool = False
    bin_size: int = 100_000
    salt: str = "entexbert2_v1"                                     # version-deterministic hashing
    fold_assignment: Dict[str, int] = field(default_factory=dict)   # chromosome -> fold index
    fold_id: int = 0                                                # which fold is TEST now
    train_frac_within_nontest: float = 8.0 / 9.0                    # 8:1 train:dev within non-test
    # --- pure genomic-bin 3-way split (used when enabled=True but no chromosome is held out) ---
    bin_test_frac: float = 0.0                                      # fraction of bins -> test (e.g. 0.10)
    bin_dev_frac: float = 0.0                                       # fraction of bins -> dev (e.g. 0.10); train = 1 - test - dev
    # --- boundary-leakage control: drop loci whose [pos-boundary_bp, pos+boundary_bp] window straddles a bin edge ---
    exclude_boundary: bool = False
    boundary_bp: int = 0                                            # half-window; set to max(left_bp, right_bp) + jitter_max_bp

    def __post_init__(self):
        if self.exclude_boundary and self.boundary_bp <= 0:
            raise ValueError("exclude_boundary=True requires boundary_bp > 0 -- (set it to max(left_bp, right_bp) of your WindowSpec)")
        if self.enabled and not self.fold_assignment and self.bin_test_frac <= 0:
            raise ValueError(
                "PartitionSpec(enabled=True) would produce no test set: supply fold_assignment "
                "(whole-chromosome hold-out) or bin_test_frac > 0 (pure genomic-bin 3-way split)")

def genomic_bin_id(chrom: str, pos: int, bin_size: int) -> str:
    """
    Deterministic genomic-bin id: chrom + which fixed-width tile the position falls in
        - uses the REFERENCE genomic position (donor-invariant) so the same locus lands in the same bin for every individual
    """
    return f"{chrom}|{int(pos) // int(bin_size)}"

def assign_split_column(df: pd.DataFrame, spec: "PartitionSpec") -> np.ndarray:
    """
    Assign a 'train'/'dev'/'test' label to every row for the hybrid partition, in two stages:
      1. a locus on a held-out test chromosome (fold_assignment[chrom] == fold_id) -> 'test'
      2. otherwise, hash its genomic bin -- int(sha1(f"{salt}|{genomic_bin_id(...)}"), 16) % 1e6 / 1e6
         and send it to 'train' if that fraction < train_frac_within_nontest, else 'dev'

    Deterministic and donor-invariant: the split depends only on (chrom, genomic position), so the
    same locus lands in the same split across every (donor, assay) cell 
    -> cross-individual train/dev vs test disjointness by construction

    Vectorized for efficiency: sha1 is evaluated once per DISTINCT genomic bin (a few 1e4 genome-wide) rather than once per row
    
    Returns an object ndarray of length len(df)
    """
    pos_col = "anchor" if "anchor" in df.columns else "ref_start"
    chroms = df["chr"].astype(str).to_numpy()
    positions = df[pos_col].astype("int64").to_numpy()

    n = len(df)
    out = np.empty(n, dtype=object)

    # Stage 1 (vectorized): loci on a held-out test chromosome -> 'test'
    test_chroms = [c for c, f in spec.fold_assignment.items() if f == spec.fold_id]
    is_test = np.isin(chroms, test_chroms) if test_chroms else np.zeros(n, dtype=bool)
    out[is_test] = "test"

    # Stage 2 (vectorized): hash each UNIQUE genomic bin once, then map the fraction back to rows
    nontest = ~is_test
    if nontest.any():
        bin_idx = positions[nontest] // int(spec.bin_size)                      # int64 tile index
        chrom_codes, chrom_uniques = pd.factorize(chroms[nontest], sort=False)
        offset = int(bin_idx.max()) + 1                                         # pack (chrom, bin) -> 1 int64
        packed = chrom_codes.astype("int64") * offset + bin_idx
        uniq, inverse = np.unique(packed, return_inverse=True)                  # (unique bins + row -> bin) map
        uniq_frac = np.empty(uniq.shape[0], dtype="float64")
        for j, val in enumerate(uniq.tolist()):
            chrom = chrom_uniques[val // offset]
            b = val % offset                                                        # so genomic_bin_id(chrom, b*bin_size, bin_size) == f"{chrom}|{b}"
            bid = genomic_bin_id(chrom, int(b) * int(spec.bin_size), spec.bin_size) # for hashing consistency within bin if bin id ever changed
            h = int(hashlib.sha1(f"{spec.salt}|{bid}".encode()).hexdigest(), 16)
            uniq_frac[j] = (h % 1_000_000) / 1_000_000.0
        fracs = uniq_frac[inverse]
        pure_bin = (len(test_chroms) == 0) and (spec.bin_test_frac > 0)
        if pure_bin:
            tf, dv = spec.bin_test_frac, spec.bin_dev_frac
            out[nontest] = np.where(
                fracs < tf, "test",
                np.where(fracs < tf + dv, "dev", "train"),
            )
        else:
            out[nontest] = np.where(fracs < spec.train_frac_within_nontest, "train", "dev")

    # Boundary exclusion: mark loci whose window straddles a bin edge for dropping
    # (applied only to hashed (nontest) rows -- whole-chromosome test sets are boundary-free by construction)
    if spec.exclude_boundary and spec.boundary_bp > 0:
        bp = int(spec.boundary_bp); bs = int(spec.bin_size)
        home = positions // bs
        same_bin = ((positions - bp) // bs == home) & ((positions + bp) // bs == home)
        out[nontest & ~same_bin] = "__exclude__"

    return out

def load_chrom_sizes(chrom_sizes_path: Optional[str]) -> Dict[str, int]:
    """
    Load chromosome sizes from a two-column chrom.sizes file
    """
    chrom_sizes = {}

    if chrom_sizes_path is None:
        return chrom_sizes

    with open(chrom_sizes_path) as f:
        for line in f:
            if not line.strip():
                continue
            chrom, size = line.strip().split()[:2]
            chrom_sizes[chrom] = int(size)

    return chrom_sizes

def split_and_write_csvs(
    df: pd.DataFrame,
    output_dir: str,
    label_col: str,
    input_mode: str = "ref_single",
    split_ratio=(0.8, 0.1, 0.1),
    seed: int = 42,
    skip_ambiguous: bool = True,
    group_cols: Optional[List[str]] = None,
    meta_cols: Optional[List[str]] = None,
    split_mode: str = "train_dev_test",
    exclude_loci: Optional[set] = None,
    dedup_sequences_across_splits: bool = True,
    balance_spec: Optional["BalanceSpec"] = None,
    balance_split: str = "all",
    split_col: Optional[str] = None,
    depth_col: Optional[str] = None,
    count_cols: Optional[List[str]] = None,
):
    """
    Write entexBERT-2 dataset CSVs

    For each split, writes two files:
      - <split>.csv       : minimal Trainer input (sequence(s), label only)
      - <split>.meta.csv  : rich superset (minimal columns + meta_cols + a 'split' column),
                            used by analysis/eval/plotting; row-aligned to <split>.csv

    Splitting:
      - split_mode == "train_dev_test": group-aware split if group_cols is provided
        (rows sharing a group key go to one split, preventing window leakage), 
        else a plain row-level shuffle split
      - split_mode == "test_only": no split; all rows written to test.csv / test.meta.csv (for scoring model on external datasets)

    exclude_loci: optional set of locus_id values to drop before writing 
    (used to build a cross-individual test set that excludes a reference model's train + dev loci)
    """
    group_cols = group_cols or []
    meta_cols = meta_cols or []

    if split_mode not in {"train_dev_test", "test_only"}:
        raise ValueError(f"Unsupported split_mode: {split_mode!r}")

    if split_mode == "train_dev_test" and not np.isclose(sum(split_ratio), 1.0):
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}.")

    paired_modes = {"hap_pair", "ref_hap1_pair", "ref_hap2_pair", "ref_alt_pair"}
    input_cols = ["sequence1", "sequence2"] if input_mode in paired_modes else ["sequence"]

    # Columns we need to carry through, de-duplicated and order-preserved
    requested = input_cols + [label_col] + group_cols + meta_cols
    if split_col is not None:
        requested = requested + [split_col]
    if depth_col is not None:
        requested = requested + [depth_col]
    if count_cols:
        requested = requested + list(count_cols)

    required = list(dict.fromkeys(requested))

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for CSV writing: {missing}")

    out = df[required].copy()
    out = out.rename(columns={label_col: "label"})
    if depth_col is not None and depth_col != "depth":
        out = out.rename(columns={depth_col: "depth"}) # rename to a stable name

    # meta_cols may reference label_col under its original name; after rename use "label"
    meta_cols = ["label" if c == label_col else c for c in meta_cols]

    if skip_ambiguous:
        if input_mode in paired_modes:
            out = out[
                ~out["sequence1"].str.contains("N", regex=False)
                & ~out["sequence2"].str.contains("N", regex=False)
            ].copy()
        else:
            out = out[~out["sequence"].str.contains("N", regex=False)].copy()

    if exclude_loci:
        if "locus_id" not in out.columns:
            raise ValueError("exclude_loci requires a 'locus_id' column (add it via meta_cols).")
        before = len(out)
        out = out[~out["locus_id"].isin(set(exclude_loci))].copy()
        print(f"exclude_loci: removed {before - len(out)} of {before} rows "
              f"({len(set(exclude_loci))} excluded loci).")

    if out.empty:
        raise ValueError("No examples remain after filtering.")

    rng = np.random.default_rng(seed) # intialize numpy random number generator

    if split_col is not None:
        _vals = set(out[split_col].astype(str).unique())
        _bad = _vals - {"train", "dev", "test"}
        if _bad:
            raise ValueError(
                f"split_col {split_col!r} has unexpected values {sorted(_bad)}; "
                f"expected only 'train'/'dev'/'test'."
            )
        splits = { 
            "train.csv": out[out[split_col] == "train"].drop(columns=[split_col]).copy(),
            "dev.csv":   out[out[split_col] == "dev"].drop(columns=[split_col]).copy(),
            "test.csv":  out[out[split_col] == "test"].drop(columns=[split_col]).copy(),
        }
        _sizes = {k: len(v) for k, v in splits.items()}
        print(f"Precomputed split via column {split_col!r}: {_sizes}")

    elif split_mode == "test_only":
        splits = {"test.csv": out.copy()}
        print(f"test_only mode: {len(out)} examples -> test.csv")

    elif group_cols:
        out["_split_group"] = out[group_cols].astype(str).agg("|".join, axis=1)

        unique_groups = out["_split_group"].drop_duplicates().to_numpy()
        rng.shuffle(unique_groups)

        n_groups = len(unique_groups)
        n_train_groups = int(split_ratio[0] * n_groups)
        n_dev_groups = int(split_ratio[1] * n_groups)

        train_groups = set(unique_groups[:n_train_groups])
        dev_groups = set(unique_groups[n_train_groups:n_train_groups + n_dev_groups])
        test_groups = set(unique_groups[n_train_groups + n_dev_groups:])

        splits = {
            "train.csv": out[out["_split_group"].isin(train_groups)].copy(),
            "dev.csv": out[out["_split_group"].isin(dev_groups)].copy(),
            "test.csv": out[out["_split_group"].isin(test_groups)].copy(),
        }

        for split_df in splits.values():
            split_df.drop(columns=["_split_group"], inplace=True)

        print(f"Group-aware split using group_cols={group_cols}")
        print(f"Unique groups: {n_groups}")
        print({
            "train_groups": len(train_groups),
            "dev_groups": len(dev_groups),
            "test_groups": len(test_groups),
        })

    else:
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)

        total = len(out)
        n_train = int(split_ratio[0] * total)
        n_dev = int(split_ratio[1] * total)

        splits = {
            "train.csv": out.iloc[:n_train].copy(),
            "dev.csv": out.iloc[n_train:n_train + n_dev].copy(),
            "test.csv": out.iloc[n_train + n_dev:].copy(),
        }

    _empty = [k for k, v in splits.items() if v.empty]
    if split_mode == "train_dev_test" and _empty:
        raise ValueError(f"empty split(s) {_empty} -- check partition_spec/split_ratio before writing.")

    # Cross-split sequence dedup: guarantee no identical input sequence appears in more than one split (priority: train > dev > test)
    if split_mode == "train_dev_test" and dedup_sequences_across_splits and len(splits) > 1:
        def _seq_key(frame):
            s = frame[input_cols[0]].astype(str)
            for c in input_cols[1:]:
                s = s + "|" + frame[c].astype(str)
            return s

        seen = set()
        total_dropped = 0
        for fn in ("train.csv", "dev.csv", "test.csv"):
            sdf = splits[fn]
            key = _seq_key(sdf)
            keep = ~key.isin(seen)
            dropped = int((~keep).sum())
            if dropped:
                total_dropped += dropped
                print(f"cross-split dedup: dropped {dropped} rows from {fn[:-4]} "
                      f"(sequence already present in an earlier split).")
            splits[fn] = sdf[keep].copy()
            seen.update(key[keep].tolist())
        if total_dropped:
            print(f"cross-split dedup: removed {total_dropped} cross-split duplicate-sequence rows total.")

    # Optionally balance ONLY the train split; dev/test stay at natural prevalence
    if (split_mode == "train_dev_test" and balance_split == "train"
            and balance_spec is not None and balance_spec.strategy != "none"
            and "train.csv" in splits and not splits["train.csv"].empty):
        from dataclasses import replace as _dc_replace
        train_df = splits["train.csv"]
        before, before_pos = len(train_df), int((train_df["label"] == 1).sum())
        splits["train.csv"] = balance_as_table(train_df, _dc_replace(balance_spec, label_col="label"))
        after = len(splits["train.csv"]); after_pos = int((splits["train.csv"]["label"] == 1).sum())
        print(f"train-only balance ({balance_spec.strategy}): train {before} -> {after} rows "
              f"(pos {before_pos} -> {after_pos}); dev/test left at natural prevalence.")

    # Save both minimal Trainer CSV columns and rich metadata CSV columns
    final_cols = input_cols + ["label"]
    if depth_col is not None:
        final_cols = final_cols + ["depth"]
    if count_cols:
        final_cols = final_cols + list(count_cols)

    meta_out_cols = list(dict.fromkeys(final_cols + meta_cols + ["split"]))

    os.makedirs(output_dir, exist_ok=True)

    for filename, split_df in splits.items():
        split_name = filename[:-len(".csv")]
        split_df = split_df.copy()
        split_df["split"] = split_name

        # Minimal training file
        split_df[final_cols].to_csv(os.path.join(output_dir, filename), index=False)

        # Rich row-aligned metadata sidecar
        meta_present = [c for c in meta_out_cols if c in split_df.columns]
        meta_name = filename.replace(".csv", ".meta.csv")
        split_df[meta_present].to_csv(os.path.join(output_dir, meta_name), index=False)

    print(f"Saved {sum(len(v) for v in splits.values())} total examples to {output_dir}")
    print({name: len(split_df) for name, split_df in splits.items()})

    combined = pd.concat([split_df[final_cols] for split_df in splits.values()], ignore_index=True)

    print("\nSequence label summary:")
    print(combined["label"].describe())

    if combined["label"].nunique() <= 20:
        print("\nSequence label counts:")
        print(combined["label"].value_counts().sort_index())

def alt_allele_of(row: pd.Series) -> Optional[str]:
    """
    The non-reference allele at a hetSNV: whichever of hap1_allele/hap2_allele differs from ref_allele;
    returns None for rows where alt is undefined
    """
    ref = str(row["ref_allele"]).upper()
    h1 = str(row["hap1_allele"]).upper()
    h2 = str(row["hap2_allele"]).upper()
    non_ref = [a for a in (h1, h2) if a != ref]
    if len(set(non_ref)) != 1: # 0 (homozygous-ref) or 2 distinct (multiallelic) -> undefined
        return None
    return non_ref[0]

def balance_as_table(
    df: pd.DataFrame,
    balance_spec: BalanceSpec,
) -> pd.DataFrame:
    """
    Balance AS examples according to a BalanceSpec
    """
    strategy = balance_spec.strategy
    label_col = balance_spec.label_col
    random_state = balance_spec.random_state

    if strategy == "none":
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if label_col not in df.columns:
        raise ValueError(f"label_col {label_col!r} not found in dataframe.")

    if strategy == "global_binary":
        pos = df[df[label_col] == 1]
        neg = df[df[label_col] == 0]

        # flag rows whose label is neither 0 nor 1 (not eligible for binary balancing)
        n_other = len(df) - len(pos) - len(neg)
        if n_other:
            _vals = pd.unique(df.loc[~df[label_col].isin([0, 1]), label_col])[:5].tolist()
            print(f"[balance] WARNING: {n_other}/{len(df)} rows ({n_other/len(df):.1%}) have "
                  f"{label_col!r} neither 0 nor 1 and are excluded from binary balancing; "
                  f"values e.g. {_vals}")

        if pos.empty or neg.empty:
            raise ValueError("Need both positive and negative examples for global_binary balancing")

        n = min(len(pos), len(neg))
        pos_sampled = pos.sample(n=n, random_state=random_state)
        neg_sampled = neg.sample(n=n, random_state=random_state)

        print(f"[balance] global_binary: {len(pos):,} pos / {len(neg):,} neg -> {n:,} each "
              f"({len(df) - 2 * n:,} of {len(df):,} rows discarded, {1 - 2 * n / len(df):.1%})")

        out = pd.concat([pos_sampled, neg_sampled], ignore_index=True)
        return out.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if strategy == "per_tissue_binary":
        if "tissue" not in df.columns:
            raise ValueError(f"per_tissue_binary balancing requires a 'tissue' column; "
                                f"have {sorted(df.columns)[:8]}.")
        balanced_groups = []

        # NOTE: technically groups by combination of tissues (since MultiTissuePeakRowSource emits pipe-joined string of tissues)
        for tissue, group in df.groupby("tissue"):
            pos = group[group[label_col] == 1]
            neg = group[group[label_col] == 0]

            if pos.empty or neg.empty:
                print(f"Skipping {tissue}: missing one class...")
                continue

            n = min(len(pos), len(neg))
            pos_sampled = pos.sample(n=n, random_state=random_state)
            neg_sampled = neg.sample(n=n, random_state=random_state)

            balanced = pd.concat([pos_sampled, neg_sampled], ignore_index=True)
            balanced_groups.append(balanced)

            print(f"{tissue}: {n} positives, {n} negatives")

        if not balanced_groups:
            raise ValueError("No tissue produced a valid balanced dataset :(")

        out = pd.concat(balanced_groups, ignore_index=True)
        return out.sample(frac=1, random_state=random_state).reset_index(drop=True)

    raise ValueError(f"Unsupported balancing strategy: {strategy}")

def add_anchor_windows(
    df: pd.DataFrame,
    window_spec: WindowSpec,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add bed_start, bed_end, anchor, and anchor_window_offset columns for anchor-centered windows
    """
    df = df.copy()

    chrom_sizes = load_chrom_sizes(window_spec.chrom_sizes_path)

    left_bp = window_spec.left_bp
    right_bp = window_spec.right_bp
    window_len = left_bp + 1 + right_bp

    mode = getattr(window_spec, "offset_mode", "fixed")
    jitter = int(getattr(window_spec, "jitter_max_bp", 0) or 0)

    if mode not in {"fixed", "uniform"}:
        raise ValueError(f"Unsupported offset_mode: {mode!r}")

    if mode == "uniform":
        if jitter < 0:
            raise ValueError("jitter_max_bp must be non-negative.")
        if jitter > min(left_bp, right_bp):
            raise ValueError(
                f"jitter_max_bp={jitter} exceeds min(left_bp, right_bp)="
                f"{min(left_bp, right_bp)}; the anchor could fall outside the window."
            )

    anchor_col = "anchor" if "anchor" in df.columns else "ref_start"
    df["anchor"] = df[anchor_col].astype(int)
    anchors = df["anchor"].to_numpy()

    rng = np.random.default_rng(seed)

    # Desired offset band (before boundary clamping)
    band_lo = left_bp - jitter if mode == "uniform" else left_bp
    band_hi = left_bp + jitter if mode == "uniform" else left_bp

    offsets = np.empty(len(df), dtype=np.int64)
    bed_starts = np.empty(len(df), dtype=np.int64)
    bed_ends = np.empty(len(df), dtype=np.int64)
    keep = np.zeros(len(df), dtype=bool)

    chroms = df["chr"].to_numpy()

    for i in range(len(df)):
        anchor = int(anchors[i])
        csize = chrom_sizes.get(chroms[i]) if chrom_sizes else None

        # Feasible offset range so the whole fixed-length window fits in the contig:
        #   bed_start = anchor - o >= 0                    -> o <= anchor
        #   bed_end   = anchor - o + window_len <= csize   -> o >= anchor + window_len - csize
        # plus keep the anchor inside the window: 0 <= o <= window_len - 1
        o_lo = max(band_lo, 0)
        o_hi = min(band_hi, window_len - 1, anchor)
        if csize is not None:
            o_lo = max(o_lo, anchor + window_len - csize)

        if o_lo > o_hi:
            keep[i] = False
            offsets[i] = -1
            bed_starts[i] = -1
            bed_ends[i] = -1
            continue

        if mode == "uniform":
            o = int(rng.integers(o_lo, o_hi + 1))
        else:
            o = left_bp # fixed; guaranteed within [o_lo, o_hi] by the check above

        offsets[i] = o
        bed_starts[i] = anchor - o
        bed_ends[i] = anchor - o + window_len
        keep[i] = True

    df["bed_start"] = bed_starts
    df["bed_end"] = bed_ends
    df["anchor_window_offset"] = offsets

    n_drop = int((~keep).sum())
    if n_drop:
        print(
            f"add_anchor_windows: dropped {n_drop} of {len(df)} windows that did not fit "
            f"the contig at the requested offset (mode={mode}, jitter={jitter})."
        )

    df = df[keep].reset_index(drop=True)

    if df.empty:
        raise ValueError("No windows remain after fitting fixed-length windows to contigs :(")

    return df

def add_label_columns(
    df: pd.DataFrame,
    primary_label: LabelSpec,
) -> pd.DataFrame:
    """
    Add the label column to the AS dataframe
    """
    df = df.copy()

    values = []
    for _, row in df.iterrows():
        raw = primary_label.fn(row)
        values.append(primary_label.transform_fn(raw))
    df[primary_label.name] = values

    # Drop rows whose primary label is undefined (NaN)
    before = len(df)
    df = df[pd.notna(df[primary_label.name])].copy()
    dropped = before - len(df)
    if dropped:
        print(f"add_label_columns: dropped {dropped}/{before} rows ({dropped/before:.1%}) with "
              f"undefined primary label '{primary_label.name}'.")
    if df.empty:
        raise ValueError(f"every one of {before} rows has an undefined primary label '{primary_label.name}'"
                         f"-- check the LabelSpec fn and its required_columns, and that the source actually provides them.")

    return df

def make_haplotype_sequence(
    ref_sequence: str,
    snv_offset: int,
    allele: str,
) -> str:
    """
    Replace the SNV position in a reference sequence with the requested allele (requires allele is a single base)
    """
    ref_sequence = ref_sequence.upper()
    allele = str(allele).upper()

    if len(allele) != 1:
        raise ValueError(f"Only SNVs are supported, got allele={allele!r}.")

    if snv_offset < 0 or snv_offset >= len(ref_sequence):
        raise ValueError(
            f"snv_offset={snv_offset} out of bounds for sequence length {len(ref_sequence)}."
        )

    return ref_sequence[:snv_offset] + allele + ref_sequence[snv_offset + 1:]

def add_sequence_inputs(
    df: pd.DataFrame,
    ref_fasta,
    input_mode: str = "ref_single",
    sequence_mode: str = "reference",
    personal_genomes: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Add sequence columns according to input_mode, plus per-sequence anchor/extent metadata

    Requires ref_fasta to support:
        str(ref_fasta[chrom][start:end])
    as pyfaidx.Fasta does

    Metadata columns added (0-based, in the stored sequence-string space):
        anchor_offset_seq1, feat_start_seq1, feat_end_seq1   (always)
        anchor_offset_seq2, feat_start_seq2, feat_end_seq2   (paired modes only)
        feature_type                                         (e.g. "snv")

    NOTE (substitution-only engine): make_haplotype_sequence currently enforces single-base alleles,
    so every sequence has identical length and the anchor offset is the same in seq1 and seq2
    """

    single_modes = {"ref_single", "hap1_single", "hap2_single"}
    paired_modes = {"hap_pair", "ref_hap1_pair", "ref_hap2_pair", "ref_alt_pair"}

    if input_mode not in single_modes and input_mode not in paired_modes:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    needs_hap1 = input_mode in {"hap1_single", "hap_pair", "ref_hap1_pair"}
    needs_hap2 = input_mode in {"hap2_single", "hap_pair", "ref_hap2_pair"}

    if sequence_mode not in {"reference", "personal"}:
        raise ValueError(f"sequence_mode must be 'reference' or 'personal', got {sequence_mode!r}.")
    if sequence_mode == "personal":
        if not personal_genomes:
            raise ValueError("sequence_mode='personal' requires personal_genomes={donor: PersonalGenome}.")
        if not (needs_hap1 or needs_hap2):
            raise ValueError(f"sequence_mode='personal' needs a haplotype input_mode "
                             f"(got {input_mode!r}, which extracts no haplotype).")
        if "donor" not in df.columns and len(personal_genomes) > 1:
                raise ValueError(f"sequence_mode='personal' with {len(personal_genomes)} donors requires a "
                                    f"'donor' column; otherwise every row would be lifted through an arbitrary genome.")

    has_alleles = ("hap1_allele" in df.columns) and ("hap2_allele" in df.columns)
    if (needs_hap1 or needs_hap2) and not has_alleles:
        raise ValueError(
            f"input_mode={input_mode!r} needs haplotype alleles, but the row source provides "
            f"no hap1_allele/hap2_allele columns (has_variants=False)."
        )

    # Ensure valid alleles
    _allele_cols = [c for c, need in (("hap1_allele", needs_hap1), ("hap2_allele", needs_hap2)) if need]
    if input_mode == "ref_alt_pair":
        _allele_cols = ["hap1_allele", "hap2_allele"]
    if _allele_cols:
        _bad = pd.Series(False, index=df.index)
        for _c in _allele_cols:
            _bad |= ~df[_c].astype(str).str.upper().isin(list("ACGT"))
        if _bad.any():
            _n = int(_bad.sum())
            raise ValueError(
                f"{_n}/{len(df)} rows ({_n/len(df):.1%}) carry a non-single-base allele in {_allele_cols} "
                f"-- only SNVs are supported. Examples: "
                f"{df.loc[_bad, _allele_cols].head(3).to_dict('records')}")

    has_ref_allele = "ref_allele" in df.columns
    has_donor_col = "donor" in df.columns

    df = df.copy()

    # ref_alt_pair builds a REFERENCE-vs-ALTERNATE contrast (sequence1=ref, sequence2=alt)
    if input_mode == "ref_alt_pair":
        if not (has_alleles and has_ref_allele):
            raise ValueError("ref_alt_pair needs ref_allele + hap1_allele/hap2_allele columns.")
        alt_defined = df.apply(lambda r: alt_allele_of(r) is not None, axis=1)
        n_drop = int((~alt_defined).sum())
        if n_drop:
            print(f"ref_alt_pair: dropping {n_drop} rows with undefined alt (homozygous-ref / multiallelic).")
        df = df[alt_defined].copy()

    sequences = []
    sequence1s = []
    sequence2s = []

    anchor1s, fstart1s, fend1s = [], [], []
    anchor2s, fstart2s, fend2s = [], [], []

    keep_pos = []
    n_dropped_personal = 0 # dropped rows in reference -> personal liftover
    _stats_before = ({d: dict(g.stats) for d, g in personal_genomes.items()}
                     if sequence_mode == "personal" else None)

    _refmm = [] # ref allele mismatch examples

    for pos, (_, row) in enumerate(df.iterrows()):
        chrom = row["chr"]
        start = int(row["bed_start"])
        end = int(row["bed_end"])
        snv = int(row["anchor"])
        a2 = s2 = e2 = None
        n_dropped_personal = 0 # dropped rows in reference -> personal liftover
        n_no_genome = 0 # rows whose donor has no PersonalGenome (a CONFIG error, not a lift failure)

        ref_seq = str(ref_fasta[chrom][start:end]).upper()

        # Sanity check that the anchor is in the provided window (should be by construction)
        if snv < start or snv >= end:
            raise ValueError(
                f"Anchor {chrom}:{snv} is outside extracted window {chrom}:{start}-{end}."
            )

        snv_offset = snv - start

        if len(ref_seq) != end - start:
            raise ValueError(
                f"Extracted sequence length mismatch at {chrom}:{start}-{end}. "
                f"Expected {end - start}, got {len(ref_seq)}."
            )

        ref_allele = str(row["ref_allele"]).upper() if has_ref_allele else None

        # Sanity check: confirm the reference FASTA base matches the table's ref allele
        if ref_allele is not None and len(ref_allele) == 1:
            observed_ref_base = ref_seq[snv_offset].upper()
            if observed_ref_base != ref_allele:
                _refmm.append((chrom, snv, observed_ref_base, ref_allele))

        hap1_allele = str(row["hap1_allele"]).upper() if has_alleles else None
        hap2_allele = str(row["hap2_allele"]).upper() if has_alleles else None

        if sequence_mode == "personal": # windows will contain all personal variants
            donor = str(row["donor"]) if has_donor_col else next(iter(personal_genomes))
            pg = personal_genomes.get(donor)
            if pg is None:
                n_dropped_personal += 1
                continue
            res, _err = pg.pair_windows(chrom, snv, hap1_allele, hap2_allele,
                                        left=snv_offset, right=(end - snv - 1))
            if res is None:
                n_dropped_personal += 1 # unliftable (variant in a gap) or allele-mismatched loci dropped
                continue
            hap1_seq = res["seq1"] if needs_hap1 else None
            hap2_seq = res["seq2"] if needs_hap2 else None
            if pg is None:
                n_no_genome += 1
                continue
        else:
            hap1_seq = make_haplotype_sequence(ref_seq, snv_offset, hap1_allele) if needs_hap1 else None
            hap2_seq = make_haplotype_sequence(ref_seq, snv_offset, hap2_allele) if needs_hap2 else None

        # Per-sequence offset/extent:
        # Substitution-only, so the anchor offset is snv_offset in every sequence;
        # extent length is the length of that sequence's allele (1 for a variant-free anchor such as a peak summit)
        def offset_extent(allele):
            ext = max(1, len(allele)) if allele else 1
            return snv_offset, snv_offset, snv_offset + ext

        if input_mode == "ref_single":
            sequences.append(ref_seq)
            a1, s1, e1 = offset_extent(ref_allele)

        elif input_mode == "hap1_single":
            sequences.append(hap1_seq)
            a1, s1, e1 = offset_extent(hap1_allele)

        elif input_mode == "hap2_single":
            sequences.append(hap2_seq)
            a1, s1, e1 = offset_extent(hap2_allele)

        elif input_mode == "hap_pair":
            sequence1s.append(hap1_seq)
            sequence2s.append(hap2_seq)
            a1, s1, e1 = offset_extent(hap1_allele)
            a2, s2, e2 = offset_extent(hap2_allele)

        elif input_mode == "ref_hap1_pair":
            sequence1s.append(ref_seq)
            sequence2s.append(hap1_seq)
            a1, s1, e1 = offset_extent(ref_allele)
            a2, s2, e2 = offset_extent(hap1_allele)

        elif input_mode == "ref_hap2_pair":
            sequence1s.append(ref_seq)
            sequence2s.append(hap2_seq)
            a1, s1, e1 = offset_extent(ref_allele)
            a2, s2, e2 = offset_extent(hap2_allele)

        elif input_mode == "ref_alt_pair":
            alt_allele = alt_allele_of(row)  # not None: undefined-alt rows dropped above
            alt_seq = make_haplotype_sequence(ref_seq, snv_offset, alt_allele)
            sequence1s.append(ref_seq)
            sequence2s.append(alt_seq)
            a1, s1, e1 = offset_extent(ref_allele)
            a2, s2, e2 = offset_extent(alt_allele)

        else:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

        anchor1s.append(a1)
        fstart1s.append(s1)
        fend1s.append(e1)

        if input_mode in paired_modes:
            anchor2s.append(a2)
            fstart2s.append(s2)
            fend2s.append(e2)
        
        keep_pos.append(pos)

    # Error if any examples where ref allele did not match expected
    if _refmm:
        _n = len(_refmm); _frac = _n / max(len(df), 1)
        _c, _p, _obs, _exp = _refmm[0]
        raise ValueError(
            f"reference allele mismatch on {_n}/{len(df)} rows ({_frac:.1%}); "
            f"~75% suggests an off-by-one (check the coordinate convention), ~100% suggests "
            f"strand-flipped alleles, a few % is expected. "
            f"First: {_c}:{_p} FASTA={_obs!r} table ref_allele={_exp!r}.")

    # personal mode may have dropped rows (unliftable / allele-mismatch); report + subset so the
    # sequence lists (built only for kept rows) align with df before the column assignments
    if sequence_mode == "personal":
        tot = sum(g.stats["total"] - _stats_before[d].get("total", 0) for d, g in personal_genomes.items())
        ok  = sum(g.stats["ok"]    - _stats_before[d].get("ok", 0)    for d, g in personal_genomes.items())
        rate = (ok / tot) if tot else float("nan")

        print(f"[personal] kept {len(keep_pos)}/{len(df)} rows "
              f"(dropped {n_dropped_personal} unliftable/allele-mismatch); "
              f"allele-match rate={rate:.3f} (low => hetSNV vs vcf2diploid reference mismatch).")
        if n_no_genome:
            _seen = sorted(set(df["donor"].astype(str)) - set(personal_genomes)) if has_donor_col else []
            print(f"[personal] WARNING: {n_no_genome} further rows dropped because their donor has "
                  f"no personal genome (donors in data but not in personal_genomes: {_seen}; "
                  f"available: {sorted(personal_genomes)}). This is a CONFIG error, not a lift "
                  f"failure -- it is excluded from the allele-match rate above.")

    if len(keep_pos) < len(df):
        df = df.iloc[keep_pos].reset_index(drop=True)

    if input_mode in single_modes:
        df["sequence"] = sequences
    elif input_mode in paired_modes:
        df["sequence1"] = sequence1s
        df["sequence2"] = sequence2s

    df["anchor_offset_seq1"] = anchor1s
    df["feat_start_seq1"] = fstart1s
    df["feat_end_seq1"] = fend1s

    if input_mode in paired_modes:
        df["anchor_offset_seq2"] = anchor2s
        df["feat_start_seq2"] = fstart2s
        df["feat_end_seq2"] = fend2s

    # Preserve a source-provided feature_type; default to "snv" for the AS/SNV source
    if "feature_type" not in df.columns:
        df["feature_type"] = "snv"

    return df

def summarize_duplicate_as_windows(
    df: pd.DataFrame,
    label_col: str = "imbalance_significance",
    group_cols: Optional[List[str]] = None,
):
    """
    Print a summary of duplicate sequence/window groups and label conflicts

    For all-tissue AS classification, duplicates often arise because the same SNV/window appears in multiple tissues
    """
    if group_cols is None:
        # emit locus_id to ensure jitter invariance
        group_cols = ["locus_id"] if "locus_id" in df.columns else [
            "chr", "anchor", "ref_allele", "hap1_allele", "hap2_allele",
        ]

    missing = [c for c in group_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for duplicate summary: {missing}")

    tmp = df.copy()
    tmp["_group_key"] = tmp[group_cols].astype(str).agg("|".join, axis=1)

    grouped = tmp.groupby("_group_key")[label_col].agg(
        n_rows="count",
        n_unique_labels="nunique",
        mean_label="mean",
        min_label="min",
        max_label="max",
    )

    duplicate_groups = grouped[grouped["n_rows"] > 1]
    conflicting_groups = grouped[grouped["n_unique_labels"] > 1]

    print("\nDuplicate AS/window summary")
    print(f"Rows: {len(tmp)}")
    print(f"Unique groups: {len(grouped)}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Rows in duplicate groups: {int(duplicate_groups['n_rows'].sum()) if len(duplicate_groups) else 0}")
    print(f"Conflicting-label groups: {len(conflicting_groups)}")
    print(f"Rows in conflicting-label groups: {int(conflicting_groups['n_rows'].sum()) if len(conflicting_groups) else 0}")

    if len(conflicting_groups) > 0:
        print("\nExample conflicting groups:")
        print(conflicting_groups.head(10))

    return grouped

def add_locus_and_example_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a jitter-invariant locus_id and a per-row-unique example_id

    - locus_id = sha1(chr|SNV)[:16]: the grouping / leakage / cross-individual-exclusion key;
      all rows for the same SNV locus (across tissues and across jitter draws) share it

    - example_id = "<chr>:<SNV>:<tissue>:<occurrence>": unique per row within a (donor, assay)
      dataset
      * NOTE: donor is NOT in the key, so rows that differ only by donor collide and are separated by <occurrence>
    """
    df = df.copy()

    df["locus_id"] = [
        hashlib.sha1(f"{c}|{int(s)}".encode()).hexdigest()[:16]
        for c, s in zip(df["chr"], df["anchor"])
    ]

    tissue = df["tissue"].astype(str) if "tissue" in df.columns else pd.Series(["NA"] * len(df), index=df.index)
    base = df["chr"].astype(str) + ":" + df["anchor"].astype(str) + ":" + tissue
    occ = base.groupby(base).cumcount().astype(str)
    df["example_id"] = base + ":" + occ

    return df

#########################
# Label-spec constructors
#########################

def make_column_label_spec(
    column: str,
    name: Optional[str] = None,
    task_type: str = "regression",
    transform_fn: Callable[[float], float] = identity_transform,
) -> LabelSpec:
    """Label read directly from a precomputed column on the row source"""
    col = column
    return LabelSpec(
        name=name or column,
        fn=lambda row: row[col],
        task_type=task_type,
        transform_fn=transform_fn,
        required_columns=[col],
    )

def summarize_bigwig_values(
    values,
    mode: str = "mean",
    missing_value: float = 0.0,
) -> float:
    """
    Summarize per-base BigWig values

    Args:
        values:
            List/array of values from pyBigWig.values(...)
            Missing bases are usually NaN
        mode:
            One of:
                - "mean"
                - "max"
                - "min"
                - "sum"
                - "std"
                - "coverage"
                - "mean_nonzero"
                - "max_abs"
        missing_value:
            Value used when no finite values are available

    Returns:
        float summary value
    """
    arr = np.asarray(values, dtype=float)

    finite = np.isfinite(arr)
    finite_vals = arr[finite]

    if mode == "coverage":
        if len(arr) == 0:
            return 0.0
        return float(finite.sum() / len(arr))

    if finite_vals.size == 0:
        return float(missing_value)

    if mode == "mean":
        return float(np.mean(finite_vals))

    if mode == "max":
        return float(np.max(finite_vals))

    if mode == "min":
        return float(np.min(finite_vals))

    if mode == "sum":
        return float(np.sum(finite_vals))

    if mode == "std":
        return float(np.std(finite_vals))

    if mode == "mean_nonzero":
        nonzero = finite_vals[finite_vals != 0]
        if nonzero.size == 0:
            return float(missing_value)
        return float(np.mean(nonzero))

    if mode == "max_abs":
        return float(np.max(np.abs(finite_vals)))

    raise ValueError(f"Unsupported BigWig signal summary mode: {mode}")

class BigWigSignalAnnotator:
    """
    Annotator for extracting continuous signal from a BigWig file

    Use for targets like:
        - fold-change-over-control signal at the SNV
        - mean signal over the sequence window
        - max signal over the sequence window
        - local signal around the SNV

    Coordinates are expected to be 0-based half-open, matching BED-style coordinates and pyBigWig queries
    """

    def __init__(
        self,
        bigwig_path: str,
        mode: str = "mean",
        region: str = "window",
        radius_bp: int = 0,
        missing_value: float = 0.0,
        use_values: bool = True,
        exact: bool = True,
    ):
        """
        Args:
            bigwig_path:
                Path to BigWig file.
            mode:
                Signal summary mode. Supported:
                    "mean", "max", "min", "sum", "std", "coverage",
                    "mean_nonzero", "max_abs"
            region:
                Region to query:
                    - "window": use row["bed_start"] to row["bed_end"]
                    - "snv": use one base [SNV, SNV + 1)
                    - "snv_radius": use [SNV - radius_bp, SNV + radius_bp + 1)
            radius_bp:
                Used only when region="snv_radius"
            missing_value:
                Returned when no finite BigWig signal exists in the queried region AND when the
                row's chromosome is absent from the BigWig
                -- pass np.nan if you want such rows dropped instead of trained on as 0's
            use_values:
                If True, use per-base bw.values(...) and summarize manually
            exact:
                If use_values=False, pass exact=exact to bw.stats(...)
        """
        if pyBigWig is None:
            raise ImportError(
                "pyBigWig is required for BigWigSignalAnnotator! "
                "Install with `pip install pyBigWig` or "
                "`conda install pybigwig -c conda-forge -c bioconda`"
            )

        self.bigwig_path = bigwig_path
        self.mode = mode
        self.region = region
        self.radius_bp = radius_bp
        self.missing_value = missing_value
        self.use_values = use_values
        self.exact = exact

        if region not in {"window", "snv", "snv_radius"}:
            raise ValueError(f"region must be one of 'window', 'snv', 'snv_radius', got {region!r}.")
        if mode not in {"mean", "max", "min", "sum", "std", "coverage", "mean_nonzero", "max_abs"}:
            raise ValueError(
                f"unsupported mode {mode!r}; choose from mean, max, min, sum, std, coverage, mean_nonzero, max_abs")
        self.bw = pyBigWig.open(bigwig_path)

        if self.bw is None:
            raise ValueError(f"Could not open BigWig file: {bigwig_path}")

        if not self.bw.isBigWig():
            self.bw.close()
            raise ValueError(f"File is not a BigWig file: {bigwig_path}")

        self.chrom_sizes = dict(self.bw.chroms())
        self._unknown_chroms = {} # chrom -> rows queried; warn once per unknown contig

    def close(self):
        if getattr(self, "bw", None) is not None:
            self.bw.close()
            self.bw = None

            def __del__(self):
                try:
                    self.close()
                except Exception:
                    pass

    def _query_region_from_row(self, row: pd.Series):
        chrom = row["chr"]

        if chrom not in self.chrom_sizes:
            n = self._unknown_chroms.get(chrom, 0) + 1
            self._unknown_chroms[chrom] = n
            if n == 1:
                print(f"[bigwig] WARNING: {chrom!r} is absent from {self.bigwig_path} -- every row "
                      f"on it returns missing_value={self.missing_value}. BigWig has e.g. "
                      f"{sorted(self.chrom_sizes)[:4]}. Check chromosome naming (chr1 vs 1).")
            return chrom, None, None

        chrom_size = self.chrom_sizes[chrom]

        if self.region == "window":
            start = int(row["bed_start"])
            end = int(row["bed_end"])

        elif self.region == "snv":
            snv = int(row["anchor"])
            start = snv
            end = snv + 1

        elif self.region == "snv_radius":
            snv = int(row["anchor"])
            start = snv - self.radius_bp
            end = snv + self.radius_bp + 1

        else:
            raise ValueError(
                f"Unsupported BigWig query region {self.region!r}. "
                "Choose from {'window', 'snv', 'snv_radius'}"
            )

        start = max(0, start)
        end = min(chrom_size, end)

        if end <= start:
            return chrom, None, None

        return chrom, start, end

    def __call__(self, row: pd.Series) -> float:
        chrom, start, end = self._query_region_from_row(row)

        if start is None or end is None:
            return float(self.missing_value)

        if self.use_values:
            values = self.bw.values(chrom, start, end)
            return summarize_bigwig_values(
                values,
                mode=self.mode,
                missing_value=self.missing_value,
            )

        # Faster option for common summary stats
        if self.mode in {"mean", "max", "min", "std", "coverage"}:
            val = self.bw.stats(
                chrom,
                start,
                end,
                type=self.mode,
                exact=self.exact, # exact=True avoids zoom-level approximations
            )[0]

            if val is None or not np.isfinite(val):
                return float(self.missing_value)

            return float(val)

        # For sum/mean_nonzero/max_abs, fall back to per-base values
        values = self.bw.values(chrom, start, end)
        return summarize_bigwig_values(
            values,
            mode=self.mode,
            missing_value=self.missing_value,
        )

def make_bigwig_label_spec(
    name: str,
    bigwig_path: str,
    mode: str = "mean",
    region: str = "window",
    radius_bp: int = 0,
    transform_fn: Callable[[float], float] = identity_transform,
    missing_value: float = 0.0,
    use_values: bool = True,
    exact: bool = True,
) -> LabelSpec:
    """
    Build a regression LabelSpec from one BigWig file
    """
    annotator = BigWigSignalAnnotator(
        bigwig_path=bigwig_path,
        mode=mode,
        region=region,
        radius_bp=radius_bp,
        missing_value=missing_value,
        use_values=use_values,
        exact=exact,
    )

    # the annotator reads region-dependent columns; build_dataset validates required_columns after windowing
    _needed = {"window":     ["chr", "bed_start", "bed_end"],
               "snv":        ["chr", "anchor"],
               "snv_radius": ["chr", "anchor"]}[region]

    return LabelSpec(
        name=name,
        fn=annotator,
        task_type="regression",
        transform_fn=transform_fn,
        required_columns=_needed,
    )

############
# RowSources
############

class RowSource:
    """
    Base class for a dataset row source

    Capability flags let build_dataset reject invalid compositions up front:
      - has_variants: whether haplotype substitution is possible
      - supported_input_modes: which sequence input_modes make sense for this source
    """
    source_type: str = "base"
    has_variants: bool = False
    supported_input_modes: frozenset = frozenset({"ref_single"})

    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"source_type": self.source_type,
                "has_variants": self.has_variants,
                "supported_input_modes": sorted(self.supported_input_modes)}

class HaplotypeCountRowSource(RowSource):
    """
    Haplotype count row source: one row per (donor, locus) with allelic read counts pooled across tissues
    (produced by build_hap_counts.py)

    Required CSV columns: chr, ref_start, ref_allele, hap1_allele, hap2_allele, k, n
        k = summed hap1 reads at the locus
        n = summed total reads (k + summed hap2 reads); supplies the per-locus depth used as the n_eff precision weight

    Optional (carried if present): imbalance_significance (the binary AS label, significant in ANY tissue), donor, assay
    """
    source_type = "hap_counts"
    has_variants = True
    supported_input_modes = frozenset({"hap_pair"})

    def __init__(
        self,
        counts_csv: str,
        assay: Optional[str] = None,
        donor: Optional[str] = None,
    ):
        self.counts_csv = counts_csv
        self.assay = assay
        self.donor = donor

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.counts_csv)
        need = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele", "k", "n"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(
                f"hap_counts source: {self.counts_csv} missing columns {missing}. "
                f"Expected the output of build_hap_counts.py. Have: {sorted(df.columns)}")

        # optional donor/assay filters
        for _attr, _col in (("donor", "donor"), ("assay", "assay")):
            _want = getattr(self, _attr)
            if _want is not None and str(_want).upper() != "ALL" and _col not in df.columns:
                raise ValueError(
                    f"hap_counts source: {_attr}={_want!r} requested but {self.counts_csv} has no "
                    f"{_col!r} column! "
                    f"Have: {sorted(df.columns)}")

        df = df.reset_index(drop=True)
        df["ref_start"] = df["ref_start"].astype(int)
        df["ref_end"] = df["ref_start"] + 1
        df["anchor"] = df["ref_start"]
        df["k"] = df["k"].astype(float)
        df["n"] = df["n"].astype(float)

        bad_mask = (df["k"] < 0) | (df["n"] < df["k"]) | (df["n"] <= 0) | df["k"].isna() | df["n"].isna()
        bad = int(bad_mask.sum())
        if bad:
            raise ValueError(
                f"hap_counts source: {bad}/{len(df)} rows ({bad/len(df):.1%}) violate 0<=k<=n, n>0 "
                f"or have missing counts. Examples: "
                f"{df.loc[bad_mask, ['chr','ref_start','k','n']].head(3).to_dict('records')}")
                
        return df

    def describe(self) -> dict:
        d = super().describe()
        d.update({"counts_csv": self.counts_csv, "assay": self.assay, "donor": self.donor})
        return d

class MultiTissuePeakRowSource(RowSource):
    """
    Multi-tissue peak row source: one row per consensus peak locus (plus sampled background)
    for Stage-1 binding supervision. Labels come from BigWig signal (not variants)

    datasets: [{tissue, peak_path, bigwig_path, [pval_path], [group]}, ...]
        group     -- optional TF name; when every entry has one, tracks are grouped by TF
                    (multi-TF joint trunk) instead of one track per tissue
        pval_path -- optional; confidence tracks are enabled only if every entry has one

    Key parameters:
        summit_mode       "summit" uses the narrowPeak summit offset; anything else uses the
                            peak midpoint
        merge_window_bp   single-linkage distance for consensus clustering: consecutive summits
                            each within this distance CHAIN into one locus, so a dense run can
                            span more than merge_window_bp
        label_radius_bp   half-width of the footprint used for label + reliability reads
        background_ratio  background loci sampled per consensus locus
        background_gap_bp minimum distance from any peak when sampling background
    """

    source_type = "multi_tissue_peak"
    has_variants = False
    supported_input_modes = frozenset({"ref_single"})

    def __init__(
        self,
        datasets: List[dict],          # [{tissue, peak_path, bigwig_path}, ...]
        assay: str,
        donor: str,
        genome_sizes_path: Optional[str] = None,
        is_narrowpeak: bool = True,    # narrowPeak input supplies signalValue
        summit_mode: str = "summit",
        merge_window_bp: int = 100,    # summits within this distance = one consensus locus
        label_radius_bp: int = 32,     # footprint for label + reliability reads (match the run)
        background_ratio: float = 1.0,
        background_gap_bp: int = 1000,
        exclude_chroms: Optional[List[str]] = None,
        seed: int = 42,
    ):
        if pyBigWig is None:
            raise ImportError("pyBigWig is required for MultiTissuePeakRowSource!")

        if not datasets:
            raise ValueError("MultiTissuePeakRowSource needs at least one {tissue, peak_path, bigwig_path} entry!")
        _req = ("tissue", "peak_path", "bigwig_path")
        for _i, _d in enumerate(datasets):
            _miss = [k for k in _req if not _d.get(k)]
            if _miss:
                raise ValueError(f"datasets[{_i}] is missing {_miss}; got keys {sorted(_d)}.")

        if genome_sizes_path is None and background_ratio > 0:
            raise ValueError(
                "background_ratio > 0 requires genome_sizes_path -- "
                "chromosome sizes are needed to sample background loci and to trim loci near contig ends. "
                "To bypass this error, input the sizes file or set background_ratio=0.")

        self.datasets = datasets
        self.assay = assay
        self.donor = donor
        self.genome_sizes_path = genome_sizes_path
        self.is_narrowpeak = is_narrowpeak

        self.summit_mode = summit_mode
        if summit_mode not in {"summit", "midpoint"}:
            raise ValueError(f"summit_mode must be 'summit' or 'midpoint', got {summit_mode!r}.")

        self.merge_window_bp = int(merge_window_bp)
        self.label_radius_bp = int(label_radius_bp)
        self.background_ratio = float(background_ratio)
        self.background_gap_bp = int(background_gap_bp)
        self.exclude_chroms = set(exclude_chroms or [])
        self.seed = int(seed)

        _groups = [d.get("group") for d in datasets]
        _n_grouped = sum(g is not None for g in _groups)
        if 0 < _n_grouped < len(_groups):
            raise ValueError(f"{_n_grouped}/{len(_groups)} datasets specify 'group' -- specify it on "
                             f"all entries (multi-TF joint trunk) or none (one track per tissue)")
        if _n_grouped == len(_groups) and _groups:  # multi-TF joint trunk
            self.track_names = list(dict.fromkeys(_groups)) # TF names
            self._group_cols = [[i for i, g in enumerate(_groups) if g == nm] for nm in self.track_names]
        else:
            self.track_names = [d["tissue"] for d in datasets]
            self._group_cols = [[i] for i in range(len(datasets))]

        # NOTE: alias "tissues" kept for callers, but in the grouped branch these are TF names (not tissues!)
        self.tissues = self.track_names

        # p-value confidence tracks are optional; enabled when every tissue provides pval_path
        self._has_pval = all(d.get("pval_path") for d in datasets)

    # ---- peak loading (per tissue) ----
    def _load_one_peakset(self, peak_path: str, tissue: str) -> pd.DataFrame:
        cols = (["chr", "start", "end", "name", "score", "strand",
                 "signalValue", "pValue", "qValue", "peak"]
                if self.is_narrowpeak else ["chr", "start", "end"])
        df = pd.read_csv(peak_path, sep="\t", header=None, comment="#",
                         usecols=range(len(cols)), names=cols)
        df = df[~df["chr"].isin(self.exclude_chroms)].copy()

        if df.empty:
                print(f"[peaks] WARNING: {tissue} contributed 0 peaks from {peak_path} "
                      f"(after excluding {sorted(self.exclude_chroms)}) -- this tissue will be "
                      f"absent from every consensus locus and from n_tissues_called")
        df["start"] = df["start"].astype(int); df["end"] = df["end"].astype(int)

        if self.summit_mode == "summit" and self.is_narrowpeak and "peak" in df:
            off = df["peak"].astype(int)
            off = off.where(off >= 0, ((df["end"] - df["start"]) // 2))
            df["summit"] = df["start"] + off
        else:
            df["summit"] = (df["start"] + df["end"]) // 2

        # BED3 input has no signalValue -> every peak gets 1.0, so the "highest signalValue" tiebreak in _consensus_loci is then just first-in-sort-order
        sv = df["signalValue"] if "signalValue" in df else pd.Series(1.0, index=df.index)
        df["tissue"] = tissue
        return df[["chr", "summit", "start", "end", "tissue"]].assign(signalValue=sv.astype(float))

    # ---- consensus clustering (sweep-line within merge_window_bp) ----
    def _consensus_loci(self, allpeaks: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for chrom, sub in allpeaks.groupby("chr"):
            sub = sub.sort_values("summit").reset_index(drop=True)
            cluster_id = (sub["summit"].diff().fillna(0) > self.merge_window_bp).cumsum()
            for _, g in sub.groupby(cluster_id):
                # canonical summit = the member with the highest signalValue
                best = g.loc[g["signalValue"].idxmax()]
                called = sorted(g["tissue"].unique().tolist())
                rows.append({
                    "chr": chrom,
                    "anchor": int(best["summit"]),
                    "pstart": int(g["start"].min()),   # widest extent (for background exclusion)
                    "pend": int(g["end"].max()),
                    "tissue": "|".join(called),
                    "n_tissues_called": len(called),
                    "called_sv_mean": float(g["signalValue"].mean()),
                })

        out = pd.DataFrame(rows)
        # NOTE: SINGLE-LINKAGE chaining -- consecutive summits each within merge_window_bp join one cluster,
        # so a dense run can span more than merge_window_bp and collapse to one anchor
        if len(out):
            span = out["pend"] - out["pstart"]
            n_wide = int((span > 10 * self.merge_window_bp).sum())
            if n_wide:
                print(f"[consensus] {n_wide}/{len(out)} loci ({n_wide/len(out):.1%}) span more than "
                      f"10x merge_window_bp (max {int(span.max()):,}bp) -- summits chained through a "
                      f"dense region; one anchor now represents that whole span.")
        return pd.DataFrame(rows)

    # ---- read all tissue BigWigs over ±radius at a set of anchors ----
    def _read_signal_matrix(self, chrom, anchors, path_key="bigwig_path") -> np.ndarray:
        """Return (n_loci, n_tissues) mean signal over [anchor-r, anchor+r+1).
        path_key selects which per-tissue track to read ('bigwig_path' = fold-change label,
        'pval_path' = signal p-value confidence). Tissues lacking path_key stay NaN."""
        r = self.label_radius_bp
        mat = np.full((len(anchors), len(self.datasets)), np.nan, dtype=float)
        for j, d in enumerate(self.datasets):
            p = d.get(path_key)
            if p is None:
                continue

            bw = pyBigWig.open(p)
            _chroms = dict(bw.chroms())
            csize = _chroms.get(chrom)
            if not csize:
                print(f"[signal] WARNING: {chrom!r} absent from {d['tissue']}'s {path_key} "
                      f"({p}) -- that track will be all-NaN (masked out) for this chromosome. "
                      f"BigWig has e.g. {sorted(_chroms)[:4]}.")

            for i, a in enumerate(anchors):
                s = max(0, a - r); e = min(csize, a + r + 1)
                if e > s:
                    v = np.asarray(bw.values(chrom, s, e), dtype=float)
                    v = v[np.isfinite(v)]
                    if v.size:
                        mat[i, j] = v.mean()
            bw.close()
        return mat

    # ---- background (single pass against the union of all peaks) ----
    def _sample_background(self, consensus: pd.DataFrame) -> pd.DataFrame:
        sizes = load_chrom_sizes(self.genome_sizes_path)
        rng = np.random.default_rng(self.seed)
        n_target = int(round(self.background_ratio * len(consensus)))
        gap = self.background_gap_bp
        forbid = {c: (sub["pstart"].to_numpy() - gap, sub["pend"].to_numpy() + gap)
                  for c, sub in consensus.groupby("chr")}
        counts = consensus["chr"].value_counts()
        chroms = counts.index.to_numpy(); weights = (counts / counts.sum()).to_numpy()
        out_c, out_a = [], []
        tries, maxt = 0, n_target * 50

        while len(out_c) < n_target and tries < maxt:
            tries += 1
            c = rng.choice(chroms, p=weights)
            csize = sizes.get(c)
            if not csize:
                continue
            a = int(rng.integers(1000, csize - 1000))
            lo, hi = forbid.get(c, (np.array([]), np.array([])))
            if lo.size and np.any((a >= lo) & (a <= hi)):
                continue
            out_c.append(c); out_a.append(a)

        out = pd.DataFrame({"chr": out_c, "anchor": out_a})
        if len(out) < n_target:
            print(f"[background] WARNING: sampled only {len(out):,}/{n_target:,} background loci "
                  f"after {tries:,} tries -- peaks+background_gap_bp={gap} leave too little free "
                  f"genome, so the effective background_ratio is "
                  f"{len(out)/max(len(consensus),1):.2f}, not {self.background_ratio}.")
        return out

    @staticmethod
    def _label_noise(cross_std: np.ndarray, mean_depth: np.ndarray) -> np.ndarray:
        """max(cross_tissue_std, depth_floor); depth_floor = c/sqrt(depth) with c
        auto-calibrated so median(floor) == median(cross_std) (commensurate units)"""
        std = np.where(np.isfinite(cross_std), cross_std, 0.0)
        dep = np.where(np.isfinite(mean_depth) & (mean_depth > 0), mean_depth, np.nan)
        med_std = np.nanmedian(std[std > 0]) if np.any(std > 0) else 1.0
        if not np.any(std > 0):
            print("[label_noise] NOTE: cross-tissue std is zero everywhere (single track?); "
                  "falling back to med_std=1.0, so the noise scale is arbitrary rather than calibrated.")
        med_dep = np.nanmedian(dep) if np.any(np.isfinite(dep)) else 1.0
        c = med_std * np.sqrt(med_dep)          # so median floor ≈ median std
        floor = np.where(np.isfinite(dep), c / np.sqrt(dep), med_std)
        return np.maximum(std, floor), float(c)

    def load(self) -> pd.DataFrame:
        # 1) union all tissue peaks, cluster into consensus loci
        allpeaks = pd.concat(
            [self._load_one_peakset(d["peak_path"], d["tissue"]) for d in self.datasets],
            ignore_index=True)
        consensus = self._consensus_loci(allpeaks)

        # drop consensus loci whose label window would fall off a chromosome end
        sizes = load_chrom_sizes(self.genome_sizes_path)
        if sizes:
            # pad by the label radius OR a generous window half-width, whichever is larger,
            # so downstream windowing (left_bp/right_bp, unknown here) can't run off the end
            r = max(self.label_radius_bp, 512)
            csz = consensus["chr"].map(sizes)
            ok = csz.notna() & (consensus["anchor"] - r >= 0) & (consensus["anchor"] + r + 1 <= csz)
            n_drop = int((~ok).sum())
            if n_drop:
                print(f"[multi_tissue] dropped {n_drop} consensus loci near chrom ends "
                      f"(window would exceed chromosome bounds)")
            consensus = consensus[ok].reset_index(drop=True)

        # 2) read all tissue BigWigs at consensus anchors -> label + reliability
        parts = []
        for chrom, sub in consensus.groupby("chr"):
            sub = sub.reset_index(drop=True)
            mat = self._read_signal_matrix(chrom, sub["anchor"].tolist())  # (n, N) fold-change
            with np.errstate(invalid="ignore"):
                _lab = np.nanmean(mat, axis=1)
                _sd  = np.nanstd(mat, axis=1)
            _n_empty = int(np.sum(~np.isfinite(_lab)))
            if _n_empty:
                print(f"[multi_tissue] {chrom}: {_n_empty}/{len(_lab)} loci had no finite signal in "
                      f"ANY track (label will be NaN) -- check BigWig chrom naming.")
            assign = dict(
                binding_label_raw=_lab,
                cross_tissue_std=_sd,
                mean_depth=sub["called_sv_mean"].to_numpy(),  # peak-call strength proxy
            )
            if self._has_pval:
                pmat = self._read_signal_matrix(chrom, sub["anchor"].tolist(), path_key="pval_path")
                assign["mean_pval"] = np.nanmean(pmat, axis=1)  # detection-confidence signal
            
            # Multi-track Stage-1: keep the PER-TISSUE fold-change vector instead of only its mean
            # y_track_t = log1p(fold-change) in tissue t; m_track_t = 1 where that tissue was actually assayed at this locus (finite), 0 where missing; mat is (n, N_tissues),
            # column order == self.tissues. NaN (unassayed) -> masked out (m=0) and set to 0.0 so
            # the value is never read when masked
            for gi, cols in enumerate(self._group_cols):
                with np.errstate(invalid="ignore"):
                    col = np.nanmean(mat[:, cols], axis=1) # merge tissues within the track/group
                m = np.isfinite(col).astype(np.int8)
                y = np.log1p(np.where(m == 1, col, 0.0))
                _bad = (m == 1) & ~np.isfinite(y)
                if _bad.any():
                    raise ValueError(
                        f"track {gi} ({self.track_names[gi]}): {int(_bad.sum())} loci have "
                        f"fold-change <= -1 (min {float(np.nanmin(col[_bad])):.3f}); log1p is "
                        f"undefined there. bigwig_path must hold NON-NEGATIVE fold-change, "
                        f"not a log-ratio.")
                assign[f"y_track_{gi}"] = y
                assign[f"m_track_{gi}"] = m
            sub = sub.assign(**assign)
            parts.append(sub)
        peaks = pd.concat(parts, ignore_index=True)
        peaks["feature_type"] = "peak"

        # 3) background once, labelled from the same BigWigs
        bg = self._sample_background(consensus)
        bgparts = []
        for chrom, sub in bg.groupby("chr"):
            sub = sub.reset_index(drop=True)
            mat = self._read_signal_matrix(chrom, sub["anchor"].tolist())
            with np.errstate(invalid="ignore"):
                _lab = np.nanmean(mat, axis=1)
                _sd  = np.nanstd(mat, axis=1)
            _n_empty = int(np.sum(~np.isfinite(_lab)))
            if _n_empty:
                print(f"[multi_tissue] background {chrom}: {_n_empty}/{len(_lab)} loci had no finite "
                      f"signal in ANY track (label will be NaN) -- check BigWig chrom naming.")
            assign = dict(
                binding_label_raw=_lab,
                cross_tissue_std=_sd,
                mean_depth=np.nan,
                tissue="background", n_tissues_called=0,
                pstart=sub["anchor"], pend=sub["anchor"] + 1,
            )
            if self._has_pval:
                pmat = self._read_signal_matrix(chrom, sub["anchor"].tolist(), path_key="pval_path")
                assign["mean_pval"] = np.nanmean(pmat, axis=1)
            # For multi-track Stage-1: per-tissue LOW signal at the background anchor
            for gi, cols in enumerate(self._group_cols):
                with np.errstate(invalid="ignore"):
                    col = np.nanmean(mat[:, cols], axis=1) # merge tissues within the track/group
                m = np.isfinite(col).astype(np.int8)
                y = np.log1p(np.where(m == 1, col, 0.0))
                _bad = (m == 1) & ~np.isfinite(y)
                if _bad.any():
                    raise ValueError(
                        f"track {gi} ({self.track_names[gi]}): {int(_bad.sum())} loci have "
                        f"fold-change <= -1 (min {float(np.nanmin(col[_bad])):.3f}); log1p is "
                        f"undefined there. bigwig_path must hold NON-NEGATIVE fold-change, "
                        f"not a log-ratio.")
                assign[f"y_track_{gi}"] = y
                assign[f"m_track_{gi}"] = m
            sub = sub.assign(**assign)
            bgparts.append(sub)
        bg = pd.concat(bgparts, ignore_index=True) if bgparts else pd.DataFrame(columns=peaks.columns)
        bg["feature_type"] = "background"

        df = pd.concat([peaks, bg], ignore_index=True)
        df["binding_label_raw"] = df["binding_label_raw"].fillna(0.0)
        if self._has_pval:
            # NaN mean_pval = no p-value coverage (background gaps) -> 0 (lowest detection onfidence);
            # optional per-locus confidence signal
            df["mean_pval"] = df["mean_pval"].fillna(0.0)
        # a locus with <2 finite tissue reads has no measurable spread -> std 0
        df["cross_tissue_std"] = df["cross_tissue_std"].fillna(0.0)
        # 4) label_noise (calibrated across ALL rows)
        ln, c = self._label_noise(df["cross_tissue_std"].to_numpy(), df["mean_depth"].to_numpy())
        df["label_noise"] = ln
        self._depth_floor_c = c

        # 5) finalize schema
        df["anchor"] = df["anchor"].astype(int)
        df["ref_start"] = df["anchor"]; df["ref_end"] = df["anchor"] + 1
        df["donor"] = self.donor; df["assay"] = self.assay
        keep = ["chr", "anchor", "ref_start", "ref_end", "donor", "tissue", "assay",
                "feature_type", "binding_label_raw",
                "n_tissues_called", "cross_tissue_std", "mean_depth", "label_noise"]
        if self._has_pval:
            keep.append("mean_pval")
        # For multi-track Stage-1: carry the per-tissue target + mask columns
        T = len(self.tissues)
        y_cols = [f"y_track_{t}" for t in range(T)]
        m_cols = [f"m_track_{t}" for t in range(T)]
        if all(c in df.columns for c in y_cols + m_cols):
            keep += y_cols + m_cols
        return df[keep].reset_index(drop=True)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"n_datasets": len(self.datasets),
                  "n_tracks": len(self.track_names),
                  "track_names": self.track_names,
                  "grouped_by": "group" if self.track_names != [x["tissue"] for x in self.datasets] else "tissue",
                  "merge_window_bp": self.merge_window_bp,
                  "label_radius_bp": self.label_radius_bp,
                  "background_ratio": self.background_ratio})
        return d

def build_dataset(
    row_source: RowSource,
    output_dir: str,
    ref_fasta,
    primary_label: LabelSpec,
    window_spec: WindowSpec,
    input_mode: str = "hap_pair",
    sequence_mode: str = "reference",
    personal_genomes: Optional[dict] = None,
    balance_spec: Optional[BalanceSpec] = None,
    balance_split: str = "all",
    split_ratio=(0.8, 0.1, 0.1),
    seed: int = 42,
    skip_ambiguous: bool = True,
    group_cols: Optional[List[str]] = None,
    split_mode: str = "train_dev_test",
    exclude_loci: Optional[set] = None,
    dedup_sequences_across_splits: bool = True,
    partition_spec: Optional["PartitionSpec"] = None,
    depth_col: Optional[str] = None,
    count_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Source-agnostic dataset builder

    Composes any RowSource with a LabelSpec, validating that:
      1. input_mode is supported by the row source
      2. the label's required_columns are provided by the source (post-windowing)

    Writes minimal Trainer CSVs + rich .meta.csv sidecars; returns the PRE-WRITE DataFrame.
    
    NOTE: the returned frame is the state just before split_and_write_csvs, so it does NOT
    reflect that function's filtering (N-containing sequences, exclude_loci, cross-split
    sequence dedup, train-only balancing)

    If partition_spec is given and enabled, a deterministic donor-invariant train/dev/test column
    is computed (hold-out-chromosome TEST + hashed genomic bins for TRAIN/DEV) and takes priority
    over the group-shuffle split
    """
    balance_spec = balance_spec or BalanceSpec(strategy="none")
    if group_cols is None:
        group_cols = ["locus_id"]

    if input_mode not in row_source.supported_input_modes:
        raise ValueError(
            f"input_mode={input_mode!r} is not supported by row source "
            f"{row_source.source_type!r} (supports {sorted(row_source.supported_input_modes)})."
        )

    df = row_source.load()
    if balance_split == "all":
        df = balance_as_table(df, balance_spec)
    elif balance_split != "train":
        raise ValueError(f"balance_split must be 'all' or 'train', got {balance_split!r}.")
    df = add_anchor_windows(df, window_spec, seed=seed)

    # Compose-time label validation (post-windowing, so window columns are available)
    needed = set(getattr(primary_label, "required_columns", []) or [])
    missing = sorted(c for c in needed if c not in df.columns)
    if missing:
        raise ValueError(
            f"Label requires columns not provided by row source "
            f"{row_source.source_type!r}: {missing}. "
            f"Available columns: {sorted(df.columns)}"
        )

    df = add_label_columns(df, primary_label=primary_label)
    df = add_sequence_inputs(df, ref_fasta=ref_fasta, input_mode=input_mode,
                             sequence_mode=sequence_mode, personal_genomes=personal_genomes)
    df = add_locus_and_example_ids(df)

    split_col = None
    if partition_spec is not None and partition_spec.enabled:
        df["_assigned_split"] = assign_split_column(df, partition_spec)
        split_col = "_assigned_split"
        _counts = pd.Series(df["_assigned_split"]).value_counts().to_dict()
        _bnd = (f"on, ±{partition_spec.boundary_bp}bp"
                if partition_spec.exclude_boundary and partition_spec.boundary_bp > 0 else "OFF")
        print(f"partition_spec enabled (bin_size={partition_spec.bin_size}, "
              f"fold_id={partition_spec.fold_id}, "
              f"boundary_exclusion={_bnd}): "
              f"row-level split counts {_counts}")

        if partition_spec.exclude_boundary:
            _need = max(window_spec.left_bp, window_spec.right_bp) + int(window_spec.jitter_max_bp or 0)
            if partition_spec.boundary_bp < _need:
                raise ValueError(
                    f"boundary_bp={partition_spec.boundary_bp} < {_need} = "
                    f"max(left_bp, right_bp) + jitter_max_bp; windows can reach {_need}bp from the anchor, "
                    f"so a smaller boundary_bp leaves edge-straddling windows in the split.")

            if (df["_assigned_split"] == "__exclude__").any():
                _n0 = len(df)
                df = df[df["_assigned_split"] != "__exclude__"].copy()
                print(f"  boundary exclusion: dropped {_n0 - len(df)} loci whose window "
                      f"straddled a {partition_spec.bin_size}bp bin edge "
                      f"(boundary_bp={partition_spec.boundary_bp})")

    if group_cols:
        summarize_duplicate_as_windows(
            df,
            label_col=primary_label.name,
            group_cols=group_cols,
        )

    paired = input_mode in {"hap_pair", "ref_hap1_pair", "ref_hap2_pair", "ref_alt_pair"}
    meta_cols = [
        "example_id", "locus_id", "feature_type",
        "anchor_offset_seq1", "feat_start_seq1", "feat_end_seq1",
    ]
    if paired:
        meta_cols += ["anchor_offset_seq2", "feat_start_seq2", "feat_end_seq2"]
    meta_cols += ["chr", "anchor", "ref_allele", "hap1_allele", "hap2_allele",
                  "tissue", "donor", "assay"]
    # Carry the AS-call columns so a regression run can be mapped back to imbalance_significance
    # post-hoc (threshold |prediction| -> AUPRC vs the binary call), and depth for stratification
    meta_cols += ["imbalance_significance", "ref_allele_ratio", "total_reads",
                  "k", "n_tissues", "n_as_reads", "weight_depth"]

    # Binding-regression reliability columns (multi_tissue_peak); skipped when absent
    meta_cols += ["binding_label_raw", "n_tissues_called", "cross_tissue_std",
                  "mean_depth", "label_noise"]
    
    meta_cols = [c for c in meta_cols if c in df.columns]

    if balance_split == "all" and balance_spec.strategy != "none":
        _lab = df[primary_label.name]
        _p, _n = int((_lab == 1).sum()), int((_lab == 0).sum())
        if _p and _n and abs(_p / _n - 1.0) > 0.05:
            print(f"[balance] NOTE: balanced to 1:1 BEFORE windowing/labelling/sequence extraction, "
                  f"but row drops since then leave {_p:,} pos / {_n:,} neg ({_p/_n:.2f} ratio) in the "
                  f"written data. Use balance_split='train' to balance after all drops.")

    split_and_write_csvs(
        df=df,
        output_dir=output_dir,
        label_col=primary_label.name,
        input_mode=input_mode,
        split_col=split_col,
        split_ratio=split_ratio,
        seed=seed,
        skip_ambiguous=skip_ambiguous,
        group_cols=group_cols,
        meta_cols=meta_cols,
        split_mode=split_mode,
        exclude_loci=exclude_loci,
        dedup_sequences_across_splits=dedup_sequences_across_splits,
        balance_spec=balance_spec,
        balance_split=balance_split,
        depth_col=depth_col,
        count_cols=count_cols,
    )

    return df