#!/usr/bin/env python3
"""
entexbert2.build_hap_counts

Use: aggregate EN-TEx hetSNV read counts per (donor, locus) via tissue pooling
into a per-locus haplotype-count table for the Stage-2 ASB head

Emitted columns (one row per donor-locus):
    chr, ref_start, ref_allele, hap1_allele, hap2_allele, donor,
    k (= summed hap1 reads), n (= summed total reads), imbalance_significance (any-sig),
    assay

The Stage-2 ASB head reads this table via the hap_counts row source where
    * n = the per-locus depth (privileged precision weight n_eff)
    * imbalance_significance = the binary AS label
    * chr/ref_start are for the leakage split

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""

import argparse, re
import numpy as np, pandas as pd

_BASECOL = {"A": "cA", "C": "cC", "G": "cG", "T": "cT"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hetsnv_tsv", required=True)
    ap.add_argument("--assay", default="CTCF", help="assay substring filter; 'ALL' keeps all")
    ap.add_argument("--min_total_reads", type=int, default=1,
                    help="drop aggregated loci with n < this (n=0 has no likelihood)")
    ap.add_argument("--out", default="hap_counts.csv")
    ap.add_argument("--per_tissue_out", default=None,
                    help="optional: also write a per-(donor,tissue,locus) companion table "
                         "(NOT a training input; for later tissue-resolved analysis)")
    a = ap.parse_args()

    usecols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
               "donor", "tissue", "assay", "cA", "cC", "cG", "cT",
               "imbalance_significance"]
               
    # chunked read + per-chunk assay filter
    n_rows = 0
    keep = []
    for ch in pd.read_csv(a.hetsnv_tsv, sep="\t", usecols=lambda c: c in usecols, chunksize=1_000_000):
        n_rows += len(ch)
        if a.assay and a.assay.upper() != "ALL":
            # fix unwanted prefix match (POLR2A and POLR2AphosphoS5)
            pat = re.escape(a.assay) + r"(?![A-Za-z0-9])"
            ch = ch[ch["assay"].astype(str).str.contains(pat, case=False, na=False, regex=True)]
        if len(ch):
            keep.append(ch)
    df = (pd.concat(keep, ignore_index=True) if keep
          else pd.DataFrame(columns=usecols)).reset_index(drop=True)

    def _counts_for(d, allele_col):
        """
        Per-row read count for the allele in `allele_col`:
        index the (N,4) cA/cC/cG/cT block by that allele's column position
        ** non-ACGT alleles -> 0.0 **
        """
        block = d[["cA", "cC", "cG", "cT"]].to_numpy(dtype=float)
        code = d[allele_col].astype(str).str.upper().map({"A": 0, "C": 1, "G": 2, "T": 3})
        ok = code.notna().to_numpy()
        idx = code.fillna(0).astype(int).to_numpy()
        vals = np.where(ok, block[np.arange(len(d)), idx], 0.0)
        return np.nan_to_num(vals, nan=0.0)

    df["hap1_count"] = _counts_for(df, "hap1_allele")
    df["hap2_count"] = _counts_for(df, "hap2_allele")
    df["imbalance_significance"] = df["imbalance_significance"].astype(int)

    # count AS reads for this tissue ONLY when this tissue itself called AS
    df["as_reads"] = (df["hap1_count"] + df["hap2_count"]) * (df["imbalance_significance"] == 1)

    # tissue pool to form aggregated (donor, locus) entries
    key = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele", "donor"]
    if a.assay and a.assay.upper() == "ALL":
        key = key + ["assay"] # perpetuate the 'ALL' key
    agg = (df.groupby(key, observed=True)
             .agg(k=("hap1_count", "sum"),
                  hap2_reads=("hap2_count", "sum"),
                  imbalance_significance=("imbalance_significance", "max"),
                  n_as_reads=("as_reads", "sum"),
                  n_tissues=("tissue", "nunique"))
             .reset_index())
    
    if "assay" not in agg.columns:
        agg["assay"] = a.assay # specific assay -- not 'ALL'

    agg["k"] = agg["k"].round().astype(int)
    agg["n"] = (agg["k"] + agg["hap2_reads"].round().astype(int)).astype(int)
    agg = agg.drop(columns="hap2_reads")

    n0 = len(agg)
    agg = agg[agg["n"] >= a.min_total_reads].reset_index(drop=True)
    agg["total_reads"] = agg["n"]
    agg["ref_allele_ratio"] = np.where(agg["n"] > 0, agg["k"] / agg["n"], np.nan)
    agg["n_as_reads"] = agg["n_as_reads"].round().astype(int)
    # weight_depth = depth of the evidence for the label
    # (AS-calling-tissue reads for positives, total pooled reads for negatives)
    agg["weight_depth"] = np.where(agg["imbalance_significance"] == 1, agg["n_as_reads"], agg["n"])

    # for directional benchmarking
    agg["hap1_is_ref"] = (agg["hap1_allele"].str.upper() == agg["ref_allele"].str.upper())

    if len(agg) == 0:
        raise SystemExit(
            f"FATAL: 0 donor-loci after aggregation ({n_rows} input rows read, assay='{a.assay}'). "
            f"Nothing written. Check --assay against the file's assay column, or lower "
            f"--min_total_reads (currently {a.min_total_reads}).")

    cols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
            "donor", "k", "n", "total_reads", "ref_allele_ratio", "n_as_reads", "weight_depth",
            "imbalance_significance", "assay", "n_tissues", "hap1_is_ref"]
    agg[cols].to_csv(a.out, index=False)

    print(f"[in]  {n_rows} rows (assay={a.assay})")
    print(f"[agg] {n0} donor-loci -> {len(agg)} kept (n>={a.min_total_reads}; dropped {n0-len(agg)})")
    print(f"[pos] imbalance_significant loci: {int(agg.imbalance_significance.sum())} "
          f"({agg.imbalance_significance.mean()*100:.1f}%)")
    print(f"[k/n] median n={int(agg.n.median())}, median k/n={ (agg.k/agg.n).median():.3f}")
    print(f"[wrote] {a.out}")

    # optional tissue-resolved table for figures
    # one row per (donor, tissue, locus) with per-tissue k/n/ratio/call, joinable to the pooled table by (chr, ref_start, donor)
    if a.per_tissue_out:
        pt = df.copy()
        pt["k"] = pt["hap1_count"].round().astype(int)
        pt["n"] = (pt["k"] + pt["hap2_count"].round().astype(int)).astype(int)
        pt = pt[pt["n"] >= a.min_total_reads].reset_index(drop=True)
        pt["total_reads"] = pt["n"]
        pt["ref_allele_ratio"] = np.where(pt["n"] > 0, pt["k"] / pt["n"], np.nan)
        pt["assay"] = a.assay
        pt_cols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
                   "donor", "tissue", "assay", "k", "n", "total_reads",
                   "ref_allele_ratio", "imbalance_significance"]
        pt[pt_cols].to_csv(a.per_tissue_out, index=False)
        print(f"[per-tissue] wrote {a.per_tissue_out}: {len(pt)} donor-tissue-loci "
              f"({pt['tissue'].nunique()} tissues)")

if __name__ == "__main__":
    main()
