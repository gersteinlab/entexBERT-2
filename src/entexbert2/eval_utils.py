#!/usr/bin/env python

"""eval_utils.py -- shared helpers for post-hoc scoring and interpretability:
- hetSNV data loading
- train/test data leakage filter
- window construction
- inference setup for twin sequences input (both haplotypes for stage 2)

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""

import os
import numpy as np
import pandas as pd
from pyfaidx import Fasta

_BASECOL = {"A": "cA", "C": "cC", "G": "cG", "T": "cT"}

def _counts_for(df, allele_col):
    """Per-row read count for the allele in allele_col: index the (N,4) cA/cC/cG/cT block by that allele's column position"""
    order = ["A", "C", "G", "T"]
    block = np.nan_to_num(df[[_BASECOL[b] for b in order]].to_numpy(dtype=float), nan=0.0) # non-ACGT alleles -> 0.0
    a = df[allele_col].astype(str).str.upper().to_numpy()
    pos = np.full(len(df), -1, dtype=int)
    for i, b in enumerate(order):
        pos[a == b] = i
    return np.where(pos >= 0, block[np.arange(len(df)), np.clip(pos, 0, 3)], 0.0)

def load_hetsnv(path, assay, min_total_reads):
    """
    Load data from (pre-downloaded) EN-TEx heterozygous SNV file (http://entex.encodeproject.org/main.html/hetSNVs_pooled_AS.tsv)
    """
    usecols = ["chr", "ref_start", "ref_end", "ref_allele", "hap1_allele", "hap2_allele",
               "donor", "tissue", "assay", "cA", "cC", "cG", "cT",
               "ref_allele_ratio", "p_betabinom", "imbalance_significance"]
    # NOTE: ref_allele_ratio and p_betabinom are not currently used, but are carried through in case they are desired for any later analysis

    REQUIRED = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
                "assay", "cA", "cC", "cG", "cT", "imbalance_significance"]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in usecols)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing required column(s) {missing}; "
                       f"found {list(df.columns)}")

    if assay and assay.upper() != "ALL":
        df = df[df["assay"].astype(str).str.contains(assay, case=False, na=False)]
    df = df.reset_index(drop=True)

    n_nonsnv = int((~df["hap1_allele"].astype(str).str.upper().isin(list("ACGT")) |
                    ~df["hap2_allele"].astype(str).str.upper().isin(list("ACGT"))).sum())
    if n_nonsnv:
        print(f"[load] WARNING: {n_nonsnv} rows have non-ACGT alleles (counts will be 0)")

    df["hap1_count"] = _counts_for(df, "hap1_allele")
    df["hap2_count"] = _counts_for(df, "hap2_allele")
    df["total_reads"] = df["hap1_count"] + df["hap2_count"]
    df["signed_log_count_ratio"] = np.log2((df["hap1_count"] + 0.5) / (df["hap2_count"] + 0.5))
    if min_total_reads:
        n0 = len(df)
        df = df[df["total_reads"] >= min_total_reads].reset_index(drop=True)
        print(f"[filter] total_reads>={min_total_reads}: {len(df)}/{n0} rows kept")
    df["label"] = df["imbalance_significance"].astype(int)

    return df

def seen_bins_from_meta(coord_files, bin_size):
    seen = set()
    for path in coord_files or []:
        if not os.path.exists(path):
            print(f"[leakage] WARNING: {path} not found; skipping."); continue
        tc = pd.read_csv(path)
        chrom_col = "chr" if "chr" in tc.columns else tc.columns[0]
        pos_col = ("SNV" if "SNV" in tc.columns else "pos" if "pos" in tc.columns
                   else "anchor" if "anchor" in tc.columns else None)
        if pos_col is None:
            print(f"[leakage] {path}: no SNV/pos/anchor column ({list(tc.columns)[:6]}...); skipping.")
            continue
        before = len(seen)
        seen |= set(zip(tc[chrom_col].astype(str), (tc[pos_col].astype(int) // bin_size)))
        print(f"[leakage] {os.path.basename(path)}: +{len(seen)-before} bins, {len(seen)} seen total")
    return seen

def flag_leaky(df, seen, bin_size, chrom_col, pos_col, pos_is_1based):
    if not seen:
        return np.zeros(len(df), dtype=bool)
    if pos_is_1based:
        bins = ((df[pos_col].astype(int) - 1) // bin_size)
    else:
        bins = (df[pos_col].astype(int) // bin_size)
    pairs = list(zip(df[chrom_col].astype(str), bins))
    return np.array([b in seen for b in pairs])

def build_windows(df, ref_fasta, left_bp, right_bp, chrom_col, pos_col, refbase_col, a1_col, a2_col, pos_is_1based, drop_refmismatch=False):
    fa = Fasta(ref_fasta, sequence_always_upper=True)
    win = left_bp + 1 + right_bp
    seqs1, seqs2, keep = [], [], []
    n_oob = n_badchrom = n_refmismatch = n_badallele = 0

    for chrom, posv, ref_a, a1, a2 in zip(
        df[chrom_col], df[pos_col], df[refbase_col], df[a1_col], df[a2_col]
    ):
        if chrom not in fa:
            keep.append(False); seqs1.append(""); seqs2.append(""); n_badchrom += 1; continue

        # Reject non-single-base alleles (N/ambiguity codes, indels, NaN, empty) that would mess up window length
        a1s, a2s = str(a1).upper(), str(a2).upper()
        if a1s not in ("A","C","G","T") or a2s not in ("A","C","G","T"):
            keep.append(False); seqs1.append(""); seqs2.append(""); n_badallele += 1; continue

        p0 = (int(posv) - 1) if pos_is_1based else int(posv) # 0-based SNV position
        start = p0 - left_bp
        end = p0 + right_bp + 1 # half-open interval
        clen = len(fa[chrom])

        if start < 0 or end > clen:
            keep.append(False); seqs1.append(""); seqs2.append(""); n_oob += 1; continue
        seq = str(fa[chrom][start:end])

        if len(seq) != win:
            keep.append(False); seqs1.append(""); seqs2.append(""); n_oob += 1; continue

        center = left_bp
        if seq[center] != str(ref_a).upper():
            n_refmismatch += 1
            if drop_refmismatch:
                keep.append(False); seqs1.append(""); seqs2.append(""); continue

        seqs1.append(seq[:center] + a1s + seq[center + 1:])
        seqs2.append(seq[:center] + a2s + seq[center + 1:])
        keep.append(True)

    n_kept = sum(keep)
    frac = n_refmismatch / max(n_kept + (n_refmismatch if drop_refmismatch else 0), 1)
    print(f"[windows] built {n_kept}/{len(df)}  "
          f"(dropped: {n_oob} out-of-bounds, {n_badchrom} bad-chrom, {n_badallele} non-ACGT-allele; "
          f"hg38-base!=ref on {n_refmismatch} rows = {frac:.1%})")
    if frac > 0.05:
        print(f"[windows] WARNING: {frac:.1%} ref-base mismatch. ~75% suggests an off-by-one "
              f"(check pos_is_1based); ~100% suggests strand-flipped alleles; a few % is normal.")

    return seqs1, seqs2, np.array(keep, dtype=bool)

def score_pairs(df, seqs1, seqs2, keep, checkpoint_dir, batch_size=64, device="cuda", overrides=None, dump_embeddings=False):
    from entexbert2.model_io import run_inference # IMPORTANT so torch is not imported here unecessarily!
    df = df.loc[keep].reset_index(drop=True)
    pairs = [[s1, s2] for s1, s2 in zip(np.asarray(seqs1)[keep], np.asarray(seqs2)[keep])]
    print(f"[score] running twin inference on {len(pairs)} variants "
          f"(dump_embeddings={dump_embeddings})...")
    if dump_embeddings:
        logits, _emb, pool_ref, pool_alt, run_config = run_inference(
            checkpoint_dir, pairs, batch_size, device, overrides or {}, dump_pools=True)
    else:
        logits, _emb, run_config = run_inference(
            checkpoint_dir, pairs, batch_size, device, overrides or {})
        pool_ref = pool_alt = None
    delta = np.asarray(logits, dtype=float).reshape(len(pairs), -1)[:, 0]
    df["delta"] = delta
    df["abs_delta"] = np.abs(delta)
    return df, run_config, pool_ref, pool_alt

def pool_hetsnv_tissues(df):
    """Pool hetSNV rows per locus across all tissues to match the tissue-pooled training label"""
    keys = [k for k in ["chr", "ref_start", "ref_end", "ref_allele", "hap1_allele",
                    "hap2_allele", "donor", "assay"] if k in df.columns]
    agg = dict(hap1_count=("hap1_count", "sum"), hap2_count=("hap2_count", "sum"),
            imbalance_significance=("imbalance_significance", "max"))
    if "tissue" in df.columns:
        agg["n_tissues"] = ("tissue", "nunique")
    g = df.groupby(keys, sort=False).agg(**agg).reset_index()
    g["total_reads"] = g["hap1_count"] + g["hap2_count"]
    g["signed_log_count_ratio"] = np.log2((g["hap1_count"] + 0.5) / (g["hap2_count"] + 0.5))
    g["label"] = g["imbalance_significance"].astype(int)
    g["tissue"] = "pooled"
    return g

def dump_pools_npz(out, df, pool_ref, pool_alt, id_col):
    """Write {out}_pools.npz with the twin pooled embeddings,
    Requires df to carry: `id_col`, `label` (from load_hetsnv), and `leaky` (from flag_leaky)"""
    missing = [c for c in (id_col, "label", "leaky") if c not in df.columns]
    if missing:
        raise KeyError(f"dump_pools_npz needs column(s) {missing} on df "
                       f"(`leaky` comes from flag_leaky, `label` from load_hetsnv); got {list(df.columns)[:12]}")

    if pool_ref is None or pool_alt is None:
        raise ValueError("dump_pools_npz called without dump_embeddings=True (pools are None)")

    path = f"{out}_pools.npz"
    np.savez_compressed(
        path,
        id=df[id_col].astype(str).to_numpy(),
        pool_ref=pool_ref.astype(np.float32),
        pool_alt=pool_alt.astype(np.float32),
        label=df["label"].astype(int).to_numpy(),
        leaky=df["leaky"].astype(bool).to_numpy(),
    )

    print(f"[dump] wrote {path}  (pool_ref {pool_ref.shape}, pool_alt {pool_alt.shape})")
