#!/usr/bin/env python
"""
entexbert2.analysis.build_asb_ism_windows -- build AS/non-AS CSVs for ISM from a given eval set

Each output row carries the same twin windows the model scored (score_individual.build_windows):
  sequence1 = reference/hap1 window  (allele1 substituted at the center of a 257bp hg38 window)
  sequence2 = alt/hap2 window        (allele2 at the center; the twin partner)
  as_label  = "AS" or "nonAS"        (feature column ism_saliency filters on)
  total_reads = per-locus read depth (coverage covariate)
  chr, anchor = chromosome + 0-based variant-center coordinate  (for the position-shortcut check)

** A Stage-1 ref-embedding probe reads sequence1 (the reference window) as the single-seq input **

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""

import argparse, os, numpy as np, pandas as pd
from pyfaidx import Fasta
from entexbert2.eval_utils import load_hetsnv, build_windows, seen_bins_from_meta, flag_leaky

def load_restrict(pv_csv, drop_leaky=True):
    """
    List of allowed (chr, ref_start) coordinates from a score_individual perVariant,
    so the ISM windows are built on the leak-free held-out eval loci (rank-by-depth within them),
    making every ISM locus carry a prediction for the confusion grid
    """
    sc = pd.read_csv(pv_csv)
    
    if drop_leaky:
        if "leaky" not in sc.columns:
            raise SystemExit(f"ERROR: {pv_csv} has no 'leaky' column -- cannot build a leak-free "
                             f"allow-list. Re-run score_individual, or pass the loci another way.")
        sc = sc[sc["leaky"] == 0]

    return {(str(c), int(p)) for c, p in zip(sc["chr"], sc["ref_start"])}

def build_entex(ref_fasta, tsv, assay, donor, min_total_reads, left_bp, right_bp, 
                    pg=None, restrict=None, train_coords=None, bin_size=100000):
    """
    Build ISM windows from EN-TEx dataset
    """
    df = load_hetsnv(tsv, assay, min_total_reads)

    if donor is None:
        raise SystemExit("ERROR: --donor is required -- pass a donor id, or ALL to pool every donor.")

    if donor and str(donor).upper() != "ALL":
        if "donor" not in df.columns:
            raise SystemExit(f"ERROR: --donor {donor} given but {tsv} has no 'donor' column")
        n = len(df)
        df = df[df["donor"].astype(str) == str(donor)].reset_index(drop=True)
        print(f"[donor] {donor}: {len(df)}/{n} rows kept")
        if df.empty:
            raise SystemExit(f"ERROR: no rows for donor {donor!r} in {tsv}")
    elif pg is not None:
        raise SystemExit("ERROR: --sequence_mode personal requires --donor!")
    df["chr"] = df["chr"].astype(str)

    sig = df["imbalance_significance"]
    bad = sorted(set(pd.unique(sig.dropna())) - {0, 1})
    n_nan = int(sig.isna().sum())

    if bad or n_nan: # imbalance significance not 0 or 1
        raise ValueError(
            f"imbalance_significance must be 0 or 1; found {bad}"
            + (f" plus {n_nan} NaN" if n_nan else "")
            + f" in {tsv}.")

    # dedup to unique loci: AS if significant in ANY row; keep first alleles + max depth at locus
    g = df.groupby(["chr", "ref_start"]).agg(
        ref_allele=("ref_allele", "first"),
        hap1_allele=("hap1_allele", "first"),
        hap2_allele=("hap2_allele", "first"),
        total_reads=("total_reads", "max"),
        imbalance_significance=("imbalance_significance", "max")).reset_index()

    # inline leak filter
    if train_coords:
        seen = seen_bins_from_meta(train_coords, bin_size)
        leaky = flag_leaky(g, seen, bin_size, "chr", "ref_start", pos_is_1based=False)
        n0 = len(g)
        g = g[~leaky].reset_index(drop=True)
        print(f"[leakfilter] dropped {int(leaky.sum())} leaky; kept {len(g)}/{n0} leak-free held-out loci")

    # keep only allow-listed (leak-free, held-out) loci
    if restrict is not None:
        n = len(g)
        g = g[[(str(c), int(p)) in restrict for c, p in zip(g["chr"], g["ref_start"])]].reset_index(drop=True)
        print(f"[restrict] kept {len(g)}/{n} loci in the allow-list")

    if not train_coords and restrict is None:
        print("[leakfilter] NO LEAK FILTER APPLIED -- windows may include Stage-2 training loci!! "
              "Pass --train_coords <train.meta.csv dev.meta.csv> or --restrict_coords!")

    seq1 = seq2 = None
    n_badallele = 0

    # reference sequence input
    if pg is None:
        s1v, s2v, keep = build_windows(g, ref_fasta, left_bp, right_bp, "chr", "ref_start",
                                       "ref_allele", "hap1_allele", "hap2_allele",
                                       pos_is_1based=False)
        keep = np.asarray(keep)
        n_badallele = int((~keep).sum())
        g = g[keep].reset_index(drop=True)
        seq1 = [x for x, k in zip(s1v, keep) if k]
        seq2 = [x for x, k in zip(s2v, keep) if k]

    rows, lift_fail = [], {}

    for i, (_, r) in enumerate(g.iterrows()):
        if pg is None:
            s1, s2 = seq1[i], seq2[i]
        else: # personal: lift the locus into both haplotypes
            res, err = pg.pair_windows(str(r["chr"]), int(r["ref_start"]),
                                       r["hap1_allele"], r["hap2_allele"], left_bp, right_bp)
            if res is None:
                lift_fail[err] = lift_fail.get(err, 0) + 1
            s1, s2 = (res["seq1"], res["seq2"]) if res is not None else (None, None)
        if s1 is None:
            continue
        rows.append((s1, s2, "AS" if int(r["imbalance_significance"]) == 1 else "nonAS",
                     float(r["total_reads"]), str(r["chr"]), int(r["ref_start"])))

    if pg is not None:
        print(f"[personal] allele-match rate = {pg.match_rate():.3f} "
              f"(kept {len(rows)}/{len(g)} loci; stats {dict(pg.stats)})")
    
    if n_badallele:
        print(f"[windows] dropped {n_badallele} loci with non-single-base alleles")
    if lift_fail:
        print(f"[personal] lift failures by reason: {lift_fail}")

    return pd.DataFrame(rows, columns=["sequence1", "sequence2", "as_label", "total_reads", "chr", "anchor"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--tf_label", required=True, help="e.g. ctcf (for output filenames)")
    ap.add_argument("--hetsnv_tsv", default=None)
    ap.add_argument("--assay", default=None, help="EN-TEx assay filter, e.g. CTCF")
    ap.add_argument("--min_total_reads", type=int, default=0)
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sequence_mode", default="reference", choices=["reference", "personal"],
                    help="personal: build twin windows by lifting each locus into the donor "
                         "haplotypes (needs --hap1_fa/--hap2_fa/--mat_chain/--pat_chain)")
    ap.add_argument("--hap1_fa"); ap.add_argument("--hap2_fa")
    ap.add_argument("--mat_chain"); ap.add_argument("--pat_chain")
    ap.add_argument("--donor", default=None,
                        help="restrict the hetSNV table to this donor before pooling; required if using personal genomes")
    ap.add_argument("--train_coords", nargs="+", default=None,
                    help="Stage-2 train.meta.csv dev.meta.csv -> inline leak filter (leak-free held-out loci)")
    ap.add_argument("--bin_size", type=int, default=100000)
    ap.add_argument("--restrict_coords", default=None,
                    help="a score_individual perVariant.csv.gz: build windows ONLY on its leak-free (leaky==0) "
                         "loci, so every ISM locus is a held-out, predictable locus for the confusion grid")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    tf = a.tf_label.lower()

    if not a.hetsnv_tsv:
        raise SystemExit("ERROR: --hetsnv_tsv is required (it is the only input this builds from).")
    
    pg = None
    if a.sequence_mode == "personal":
        from entexbert2.personal_seq import PersonalGenome, parse_chain_file
        for name, val in (("hap1_fa", a.hap1_fa), ("hap2_fa", a.hap2_fa),
                            ("mat_chain", a.mat_chain), ("pat_chain", a.pat_chain)):
            if not val:
                raise SystemExit(f"--sequence_mode personal requires --{name}")
        pg = PersonalGenome(parse_chain_file(a.mat_chain), Fasta(a.hap1_fa),
                            parse_chain_file(a.pat_chain), Fasta(a.hap2_fa), donor=a.donor)
    restrict = load_restrict(a.restrict_coords) if a.restrict_coords else None

    d = build_entex(a.ref_fasta, a.hetsnv_tsv, a.assay, a.donor, a.min_total_reads,
            a.left_bp, a.right_bp, pg=pg, restrict=restrict,
            train_coords=a.train_coords, bin_size=a.bin_size)

    suffix = "_personal" if a.sequence_mode == "personal" else ""
    out = os.path.join(a.outdir, f"{tf}_asb_ism_windows_entex{suffix}.csv")
    d.to_csv(out, index=False)
    vc = d["as_label"].value_counts().to_dict()
    print(f"[entex:{a.sequence_mode}] wrote {out}: {len(d)} loci {vc}")

if __name__ == "__main__":
    main()
