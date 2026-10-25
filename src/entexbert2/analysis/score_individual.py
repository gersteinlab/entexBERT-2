#!/usr/bin/env python
"""
entexbert2.analysis.score_individual -- within-individual ASB-head scoring (both arms --ref and --personal supported)

NOTE: does NOT compute AUROC (within-individual AUROC read from the Trainer output eval_results.json)

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""
import argparse
import numpy as np
import pandas as pd
from entexbert2.eval_utils import (load_hetsnv, seen_bins_from_meta, flag_leaky,
                                    pool_hetsnv_tissues, build_windows, score_pairs, dump_pools_npz)

def eval_ref(args):
    """
    hg38 hetSNV windows -> stage 2 per-locus ell
    Tissue-pooled and leak-flagged (--drop_leaky filters to the leak-free set)
    """
    full = load_hetsnv(args.hetsnv_tsv, args.assay, args.min_total_reads)
    print(f"[load] {len(full)} hetSNV rows over donors={sorted(full.donor.unique())}")
    
    n = len(full) # number of rows before pooling
    full = pool_hetsnv_tissues(full) # pool over tissues as is done in training
    print(f"[pool] {n} rows -> {len(full)} per-locus")
    
    seen = seen_bins_from_meta(args.train_coords, args.bin_size)

    all_rows = []

    for donor in args.donors:
        d = full[full["donor"] == donor].reset_index(drop=True)

        if len(d) == 0:
            print(f"\n[{donor}] no rows; skipping..."); continue

        print(f"\n===== donor {donor}: {len(d)} rows "
              f"(pos={int((d.label == 1).sum())}, neg={int((d.label == 0).sum())}) =====")

        seqs1, seqs2, keep = build_windows(
            d, args.ref_fasta, args.left_bp, args.right_bp,
            chrom_col="chr", pos_col="ref_start", refbase_col="ref_allele",
            a1_col="hap1_allele", a2_col="hap2_allele", pos_is_1based=False)

        d, _run_config, pool_ref, pool_alt = score_pairs(
            d, seqs1, seqs2, keep, args.checkpoint_dir,
            batch_size=args.batch_size, device=args.device,
            overrides=args.overrides, dump_embeddings=args.dump_embeddings)

        d["donor"] = donor
        d["donor_kind"] = "matched" if donor == args.matched_donor else "cross-donor"
        d["leaky"] = flag_leaky(d, seen, args.bin_size, "chr", "ref_start", pos_is_1based=False)

        if seen:
            kind = d["donor_kind"].iloc[0]
            print(f"[leakage] {int(d['leaky'].sum())}/{len(d)} {donor} ({kind}) in a seen bin "
                  f"({int(d.loc[d.label == 1, 'leaky'].sum())} positive)")

        all_rows.append((d, pool_ref, pool_alt))

    if not all_rows:
        print("[warn] no rows scored!")
        return

    alld = pd.concat([r[0] for r in all_rows], ignore_index=True)

    if args.drop_leaky:
        n = len(alld); alld = alld[~alld["leaky"]].reset_index(drop=True)
        print(f"[drop_leaky] {n} -> {len(alld)} leak-free rows")

    pv_cols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
               "donor", "donor_kind", "tissue", "label", "total_reads",
               "signed_log_count_ratio", "delta", "abs_delta", "leaky"]

    pv = f"{args.out}_perVariant.csv.gz"
    alld[pv_cols].to_csv(pv, index=False, compression="gzip")
    print(f"[write] {pv}  ({len(alld)} loci) -- ell prediction for the confusion grid")

    if args.dump_embeddings:
        pr = np.concatenate([r[1] for r in all_rows], axis=0)
        pa = np.concatenate([r[2] for r in all_rows], axis=0)
        alld["_id"] = (alld["chr"].astype(str) + ":" + alld["ref_start"].astype(str)
                       + "_" + alld["donor"].astype(str))
        dump_pools_npz(args.out, alld, pr, pa, "_id")

def eval_personal(args):
    """
    Personal head on its pre-built personal twin windows -> per-locus ell
    ** leak flag is recomputed here from the hg38 anchor and --train_coords **
    NOTE: leaky rows are not dropped, just labeled (users should filter on the leaky column downstream)
    """
    from entexbert2.model_io import run_inference
    df = pd.read_csv(args.windows_csv)
    pairs = [[s1, s2] for s1, s2 in zip(df["sequence1"].astype(str), df["sequence2"].astype(str))]
    logits, _emb, _rc = run_inference(args.checkpoint_dir, pairs,
                                      batch_size=args.batch_size, device=args.device)

    delta = np.asarray(logits).reshape(-1) # ell = a*||z1-z2|| + b (classification)\

    seen = seen_bins_from_meta(args.train_coords, args.bin_size)
    leaky = flag_leaky(pd.DataFrame({"chr": df["chr"].astype(str),
                                     "ref_start": df["anchor"].astype(int)}),
                       seen, args.bin_size, "chr", "ref_start", pos_is_1based=False)
    if not seen:
        print("[leakage] WARNING: no --train_coords given -- leaky=False for every row is unverified! "
              "Pass train.meta.csv dev.meta.csv to compute it.")
    else:
        print(f"[leakage] {int(leaky.sum())}/{len(df)} personal loci in a seen bin")

    out = pd.DataFrame(dict(
        chr=df["chr"].astype(str),
        ref_start=df["anchor"].astype(int), # grid join key = hg38 0-based anchor
        delta=delta, abs_delta=np.abs(delta),
        label=(df["as_label"].astype(str).str.strip().str.upper() == "AS").astype(int),
        total_reads=df["total_reads"] if "total_reads" in df.columns else 0.0,
        donor_kind="matched", # personal -> same-donor
        leaky=leaky))

    pv = f"{args.out}_perVariant.csv.gz"
    out.to_csv(pv, index=False, compression="gzip")
    print(f"[write] {pv}  ({len(out)} loci) -- ell prediction for the confusion grid  "
          f"delta [{delta.min():.3f}, {delta.max():.3f}]  AS/nonAS {out.label.value_counts().to_dict()}")

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["ref", "personal"], required=True,
                    help="ref = hg38 hetSNV windows (built here); personal = pre-built personal windows")
    ap.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint directory (has run_config.json)")
    ap.add_argument("--out", required=True, help="output PREFIX -> {out}_perVariant.csv.gz")
    # reference
    ap.add_argument("--ref_fasta", default=None, help="hg38.fa (ref arm)")
    ap.add_argument("--hetsnv_tsv", default=None, help="hetSNVs.tsv (ref arm)")
    ap.add_argument("--assay", default="CTCF")
    ap.add_argument("--donors", nargs="+", default=["ENC-001", "ENC-002", "ENC-003", "ENC-004"])
    ap.add_argument("--matched_donor", required=True,
                    help="the donor this head was TRAINED on; rows for it are labeled "
                         "donor_kind='matched', all others labeled 'cross-donor'")
    ap.add_argument("--min_total_reads", type=int, default=20)
    ap.add_argument("--drop_leaky", action="store_true",
                    help="drop rows whose bin was seen in --train_coords -> leak-free perVariant")
    # personal
    ap.add_argument("--windows_csv", default=None, help="*_entex_personal.csv (personal arm)")
    # shared
    ap.add_argument("--train_coords", nargs="+", default=None,
                    help="the meta CSVs for the splits this head TRAINED on -- pass BOTH "
                         "train.meta.csv and dev.meta.csv. "
                         "Never pass test.meta.csv "
                         "-- that is the eval set, and it would flag every locus as leaky!")
    ap.add_argument("--bin_size", type=int, default=100000)
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overrides", default=None,
                    help="JSON dict of run_config overrides, e.g. '{\"proj_dim\": 64}'")
    ap.add_argument("--dump_embeddings", action="store_true",
                    help="also write {out}_pools.npz (raw ref/alt pools for probe scripts)")
    args = ap.parse_args()
    if args.arm == "ref":
        if not args.hetsnv_tsv:
            ap.error("--arm ref requires --hetsnv_tsv")
        if not args.ref_fasta:
            ap.error("--arm ref requires --ref_fasta")
    else:
        if not args.windows_csv:
            ap.error("--arm personal requires --windows_csv")

    if args.overrides:
        import json
        try:
            args.overrides = json.loads(args.overrides)
        except json.JSONDecodeError as e:
            ap.error(f"--overrides must be a JSON object: {e}")

    return args

def main():
    args = parse_args()
    if args.arm == "ref":
        eval_ref(args)
    else:
        eval_personal(args)

if __name__ == "__main__":
    main()
