#!/usr/bin/env python
"""
entexbert2.analysis.score_crossdonor -- cross-donor test set ASB-head scoring
(both ref and personal test sets supported)

Writes one per-head CSV: assay, train_donor, eval_donor, arm, donor_kind, auroc, n, n_pos

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def score_on(head_dir, test_csv, batch_size, device):
    from entexbert2.model_io import run_inference # to keep --help torch-free
    df = pd.read_csv(test_csv)

    if "sequence1" not in df.columns or "sequence2" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{test_csv}: need sequence1,sequence2,label; got {list(df.columns)[:6]}")

    pairs = [[s1, s2] for s1, s2 in zip(df["sequence1"].astype(str), df["sequence2"].astype(str))]
    logits, _emb, _rc = run_inference(head_dir, pairs, batch_size, device)
    ell = np.asarray(logits).reshape(-1)
    y = df["label"].astype(int).to_numpy()

    # compute AUROC for cross-donor
    auroc = float(roc_auc_score(y, ell)) if y.min() != y.max() else float("nan")
    return auroc, len(y), int(y.sum())

def _meta_coords(path):
    """(chr, pos) set from a *.meta.csv"""
    tc = pd.read_csv(path)
    chrom_col = "chr" if "chr" in tc.columns else tc.columns[0]
    pos_col = ("SNV" if "SNV" in tc.columns else "pos" if "pos" in tc.columns
               else "anchor" if "anchor" in tc.columns else
               "ref_start" if "ref_start" in tc.columns else None)
    if pos_col is None:
        raise ValueError(f"{path}: no SNV/pos/anchor/ref_start column ({list(tc.columns)[:6]}...)")
    return set(zip(tc[chrom_col].astype(str), tc[pos_col].astype(int)))

def verify_folds(exp_root, assay, arm, train_donor, donors, bin_size):
    """
    Check the crucial two cross-donor comparison conditions:

      (1) LEAKAGE: the head's train+dev coordinates must not appear in any eval donor's test set (checked within bin_size)
      (2) COMPARABILITY: every eval donor's test set should hold the SAME coordinates

    Returns True if no leakage was found
    Comparability differences warn but do not fail
    """
    def meta(d, split):
        return os.path.join(exp_root, f"{assay.lower()}_{d}", f"stage2_{arm}_data", f"{split}.meta.csv")

    seen = set()
    for split in ("train", "dev"):
        p = meta(train_donor, split)
        if not os.path.exists(p):
            print(f"[verify_folds] WARNING: {p} not found -- leak check INCOMPLETE")
            continue
        seen |= _meta_coords(p)

    if not seen:
        print("[verify_folds] WARNING: no train/dev coords for "
              f"{train_donor}; cannot verify leakage for this donor.")
        return True

    seen_bins = {(c, p // bin_size) for c, p in seen}
    print(f"[verify_folds] {train_donor} train+dev: {len(seen)} loci, {len(seen_bins)} bins")

    test_sets, leaked = {}, False
    for ed in donors:
        p = meta(ed, "test")
        if not os.path.exists(p):
            print(f"[verify_folds] {ed}: no test.meta.csv -- skipped")
            continue
        t = _meta_coords(p)
        test_sets[ed] = t
        ex = len(t & seen)
        bn = len({(c, q // bin_size) for c, q in t} & seen_bins)
        flag = "LEAK" if (ex or bn) else "clean"
        print(f"[verify_folds] {ed} test: {len(t)} loci | exact overlap {ex} | "
              f"bin overlap {bn} -> {flag}")
        if ex or bn:
            leaked = True

    if len(test_sets) > 1:
        # check that no test binds for one donor end up in train or dev for another
        seen_by = {}
        for d in test_sets:
            s = set()
            for split in ("train", "dev"):
                p = meta(d, split)
                if os.path.exists(p):
                    s |= {(c, q // bin_size) for c, q in _meta_coords(p)}
            seen_by[d] = s
        tbins = {d: {(c, q // bin_size) for c, q in t} for d, t in test_sets.items()}
        bad = {f"{d1}.test n {d2}.train/dev": len(tbins[d1] & seen_by[d2])
               for d1 in tbins for d2 in seen_by
               if d1 != d2 and seen_by[d2] and tbins[d1] & seen_by[d2]}
        if bad:
            print(f"[verify_folds] WARNING: the fold is INCONSISTENT across donors -- bins that are "
                  f"test for one donor are train/dev for another: {bad}")
        else:
            cov = {d: len(t) for d, t in test_sets.items()}
            print(f"[verify_folds] fold is consistent: no donor's test bin is any other donor's train/dev bin!")

    return not leaked

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="the trained head (train_donor's arm)")
    ap.add_argument("--assay", required=True)
    ap.add_argument("--train_donor", required=True)
    ap.add_argument("--arm", choices=["ref", "personal"], default="ref")
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--donors", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="per-head output CSV (overwritten, not appended)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify_folds", action="store_true",
                    help="before scoring, check that for cross-donor leakage and comparability")
    ap.add_argument("--bin_size", type=int, default=100000,
                    help="leak-bin size for --verify_folds; must match the scorers' bin_size")
    a = ap.parse_args()

    if a.verify_folds:
        if not verify_folds(a.exp_root, a.assay, a.arm, a.train_donor, a.donors, a.bin_size):
            raise SystemExit(
                "FATAL: the head's train/dev coordinates overlap an eval donor's test set -- "
                "the cross-donor AUROCs would be leaky! Rebuild the Stage-2 folds, or drop the "
                "offending donor from --donors.")

    rows = []

    for ed in a.donors:
        tcsv = os.path.join(a.exp_root, f"{a.assay.lower()}_{ed}", f"stage2_{a.arm}_data", "test.csv")
        if not os.path.exists(tcsv):
            print(f"[skip] {a.assay}/{ed} {a.arm}: no test.csv at {tcsv}")
            continue
        au, n, npos = score_on(a.checkpoint_dir, tcsv, a.batch_size, a.device)
        kind = "matched" if ed == a.train_donor else "cross-donor"
        rows.append(dict(assay=a.assay, train_donor=a.train_donor, eval_donor=ed, arm=a.arm,
                         donor_kind=kind, auroc=au, n=n, n_pos=npos))
        print(f"  [{a.assay}] {a.train_donor} -> {ed} ({kind}) {a.arm}: "
              f"AUROC={au:.4f} n={n} n_pos={npos}")

    if not rows:
        print("[warn] no rows produced (no test.csv found)"); return

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[write] {a.out} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
