#!/usr/bin/env python3
"""
entexbert2.select_heldout_chroms.py — deterministic, TF-agnostic held-out test chromosome selection

Pure function of a per-chromosome positive-count table plus four documented, genome-level policy parameters
Re-run on any TF's variant set to regenerate its held-out test set

Usage:
  python select_heldout_chroms.py \
      --eval_csv /path/to/file_generated_by_build_hap_counts.csv \
      --out_prefix assay_name --label_col imbalance_significance \
      [--target 0.10 --top_train_spare 2 --max_set 2 --ci_floor 100]

Outputs: <out_prefix>_per_chrom_counts.csv and <out_prefix>_heldout.json
(the chosen set plus every policy parameter, so the split can be regenerated from just the json)
"""
import json
import argparse, itertools, sys
import pandas as pd, numpy as np

# genome-level policy (TF-INDEPENDENT): special-regime chromosomes to never put in the test split
SPECIAL_DEFAULT = ["chrX", "chrY", "chr6",                       # sex + MHC/HLA
                   "chr13", "chr14", "chr15", "chr21", "chr22"]  # acrocentric (rDNA/satellite)

def natural_key(c):
    x = c.replace("chr", "")
    return (0, int(x)) if x.isdigit() else (1, x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", required=True,
                    help="variant table with a 'chr' column and a binary positive-label column")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--label_col", default="label",
                    help="name of the 0/1 positive-label column")
    ap.add_argument("--target", type=float, default=0.10,
                    help="coverage-target fraction of total positives")
    ap.add_argument("--top_train_spare", type=int, default=2,
                    help="spare the N largest chromosomes (by variant count) from holdout")
    ap.add_argument("--max_set", type=int, default=2,
                    help="max chromosomes in the held-out set")
    ap.add_argument("--ci_floor", type=int, default=100,
                    help="min positives in the chosen set; below this number, program will warn and fall back to hash-bin split only")
    ap.add_argument("--special", nargs="+", default=list(SPECIAL_DEFAULT),
                    help="special-regime chromosomes always excluded from holdout (default: sex, MHC/HLA, acrocentric)")
    a = ap.parse_args()

    df = pd.read_csv(a.eval_csv, usecols=lambda c: c in ("chr", a.label_col))
    if a.label_col not in df.columns:
        sys.exit(f"ERROR: label column {a.label_col!r} not found in {a.eval_csv}")

    if "chr" not in df.columns:
        sys.exit(f"ERROR: column 'chr' not found in {a.eval_csv} "
                 f"(columns read: {list(df.columns)})")

    df["_lab"] = df[a.label_col].astype(int)
    g = (df.groupby("chr")["_lab"].agg(n="size", pos="sum").reset_index())
    g["k"] = g["chr"].map(natural_key)
    g = g.sort_values("k").drop(columns="k").reset_index(drop=True) # sort in chr1, chr2, ..., chr23, chrX, chrY order
    
    # count positive sequences per chromosome
    tot = int(g["pos"].sum())
    if tot == 0:
        sys.exit(f"ERROR: 0 positives in column {a.label_col!r} across {len(g)} chromosomes! "
                 f"Check --label_col — is it the right column, and is it 0/1?")

    g["pos_frac"] = g["pos"] / tot
    g.to_csv(f"{a.out_prefix}_per_chrom_counts.csv", index=False)

    # Leave the N largest chromosomes by VARIANT COUNT (n) in train to preserve the best-powered data there
    by_size = g.sort_values("n", ascending=False)["chr"].tolist()
    spare_big = set(by_size[:a.top_train_spare])
    excl = set(a.special) | spare_big
    
    elig = g[~g["chr"].isin(excl)] # eligable chromosomes for test split holdout after filtering
    pos_per_elig = dict(zip(elig["chr"], elig["pos"])) # map: eligible chromosome -> positive count in chrom

    if not pos_per_elig:
        sys.exit(f"ERROR: no eligible chromosomes ({len(g)} present; "
                 f"{len(set(a.special) & set(g['chr']))} excluded as special-regime, "
                 f"{a.top_train_spare} spared as largest). "
                 f"Lower --top_train_spare or shorten --special to hold out full chromosomes for test")

    cands = []
    for r in range(1, a.max_set + 1):
        for cmb in itertools.combinations(sorted(pos_per_elig, key=natural_key), r):
            p = sum(pos_per_elig[c] for c in cmb)
            cands.append((cmb, p, p / tot))

    # pick closest to target positive fraction and tie-break by natural chromosome order to preserve determinism
    cands.sort(key=lambda x: (abs(x[2] - a.target), [natural_key(c) for c in x[0]]))
    chosen, cpos, cfrac = cands[0]

    print(f"total positives = {tot}  over {len(g)} chromosomes")
    print(f"largest-{a.top_train_spare} by variant count (spared from holdout): "
          f"{sorted(spare_big, key=natural_key)}")
    print(f"special-regime excluded: {sorted(set(a.special), key=natural_key)}")
    print(f"\nHELD-OUT SET: {'+'.join(chosen)}  pos={cpos}  frac={cfrac:.4f} "
          f"(target {a.target:.0%}, dev {abs(cfrac-a.target):.4f})")
    if cpos < a.ci_floor:
        print(f"\n!! CI-VIABILITY WARNING: {cpos} positives < floor {a.ci_floor}. "
              f"Whole-chromosome tier too sparse for this TF — report the hash-bin "
              f"leak-free number only, not a whole-chromosome CI.")
    else:
        print(f"CI-viability OK ({cpos} >= {a.ci_floor} positives)")
    print(f"\nwrote {a.out_prefix}_per_chrom_counts.csv")

    meta = {
        "chosen_heldout_chroms": list(chosen),
        "positives_in_set": int(cpos),
        "positive_fraction": float(cfrac),
        "deviation_from_target": float(abs(cfrac - a.target)),
        "total_positives": tot,
        "n_chromosomes": int(len(g)),
        "ci_viable": bool(cpos >= a.ci_floor),
        "target": a.target, "ci_floor": a.ci_floor, "max_set": a.max_set,
        "top_train_spare": a.top_train_spare,
        "spared_largest": sorted(spare_big, key=natural_key),
        "special_excluded": sorted(set(a.special), key=natural_key),
        "label_col": a.label_col, "eval_csv": a.eval_csv,
    }

    with open(f"{a.out_prefix}_heldout.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f"wrote {a.out_prefix}_heldout.json")

if __name__ == "__main__":
    main()
