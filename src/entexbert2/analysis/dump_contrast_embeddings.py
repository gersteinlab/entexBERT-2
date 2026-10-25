#!/usr/bin/env python
"""
entexbert2.analysis.dump_contrast_embeddings -- dump per-locus contrast representations

  * (Stage 1) TRUNK contrast: h1 - h2      (pool_ref - pool_alt, dimension = d_model)
        -- what the frozen trunk already produces, BEFORE the stage 2 learned projection

  * (Stage 2) HEAD contrast: P(h1) - P(h2) (dimension = proj_dim)

Also save: the model's twin classification logit ell (= a*s + b = logit P(ASB)) and the binary AS label
(if the input CSV carries total_reads / chr / anchor, they are also passed through to enable leak filtering)

Input:
a Stage-2 build CSV (columns: sequence1, sequence2, label[, depth])
OR an ASB-windows CSV (sequence1, sequence2, as_label, total_reads, chr, anchor)

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""
import argparse, csv, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="trained classification Stage-2 run directory")
    ap.add_argument("--data_csv", required=True, help="Stage-2 build CSV (sequence1,sequence2,label)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_rows", type=int, default=0,
                    help="0 = all rows; else cap number of rows by randomly subsetting negatives")
    ap.add_argument("--neg_per_pos", type=float, default=None,
                    help="negatives to keep PER POSITIVE (1.0 = balanced); overrides --max_rows if given")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_col", default="label",
                    help="label column; numeric (e.g. 0.5 for '> 0.5 => positive') OR categorical AS/nonAS (as_label)")
    args = ap.parse_args()

    from entexbert2.model_io import run_inference

    with open(args.data_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"empty CSV: {args.data_csv}")
    cols = rows[0].keys()
    for need in ("sequence1", "sequence2", args.label_col):
        if need not in cols:
            sys.exit(f"{args.data_csv} missing column '{need}'; has {list(cols)}")

    # allow for label options
    def _lab(v):
        s = str(v).strip()
        if s.lower() in ("as", "1", "true"):  return 1
        if s.lower() in ("nonas", "non-as", "non_as", "non as", "0", "false"):  return 0
        try:
            return int(float(s) > 0.5) # fallback to predicting AS if logit > 0.5, non-AS if < 0.5
        except ValueError:
            raise SystemExit(f"--label_col {args.label_col!r}: unrecognised value {v!r}; expected "
                             f"AS/nonAS/1/0/true/false or a number")

    labels = np.array([_lab(r[args.label_col]) for r in rows], dtype=int)
    
    # optional subsample to bound GPU cost
    if args.neg_per_pos is not None or (args.max_rows and args.max_rows < len(rows)):
        rng = np.random.default_rng(args.seed)
        pos, neg = idx[labels == 1], idx[labels == 0]
        if args.neg_per_pos is not None:
            n_neg_keep = int(round(args.neg_per_pos * len(pos)))
            mode = f"--neg_per_pos {args.neg_per_pos}"
        else:
            n_neg_keep = max(0, args.max_rows - len(pos))
            mode = f"--max_rows {args.max_rows}"
        neg_keep = rng.choice(neg, size=min(n_neg_keep, len(neg)), replace=False)
        idx = np.sort(np.concatenate([pos, neg_keep]))
        rows = [rows[i] for i in idx]
        labels = labels[idx]

        print(f"[subsample] {mode} -> kept {len(rows)} rows "
              f"({int((labels==1).sum())} pos, {int((labels==0).sum())} neg)")

        if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
            raise SystemExit(
                f"{mode} left a SINGLE-CLASS npz ({int((labels==1).sum())} pos, "
                f"{int((labels==0).sum())} neg) -- every consumer computes an AUROC and would "
                f"return nan. Use --neg_per_pos 1.0, or raise --max_rows above the positive count.")

    texts = [[r["sequence1"], r["sequence2"]] for r in rows]
    print(f"[in] {len(texts)} loci | pos={int((labels==1).sum())} neg={int((labels==0).sum())}")

    # carry coverage and coordinates through to the npz if present
    def _meta(name, default, cast):
        if name in cols:
            return np.array([cast(r.get(name) or default) for r in rows])
        return np.array([default] * len(rows))

    total_reads = _meta("total_reads", 0.0, float).astype(np.float32)
    chrom = _meta("chr", "?", str)
    anchor = _meta("anchor", 0, lambda v: int(float(v))).astype(np.int64)

    logits, emb, pool_ref, pool_alt, rc = run_inference(
        args.checkpoint_dir, texts, batch_size=args.batch_size,
        device=args.device, dump_pools=True)

    logits = np.asarray(logits, dtype=np.float32).reshape(-1)     # ell = logit P(ASB)
    head_contrast = np.asarray(emb, dtype=np.float32)             # (N, proj_dim)
    pool_ref = np.asarray(pool_ref, dtype=np.float32)             # (N, H)
    pool_alt = np.asarray(pool_alt, dtype=np.float32)             # (N, H)
    trunk_contrast = pool_ref - pool_alt                          # (N, H)

    assert head_contrast.shape[0] == trunk_contrast.shape[0] == len(labels), "row mismatch"
    print(f"[dump] trunk_contrast {trunk_contrast.shape} | head_contrast {head_contrast.shape} "
          f"| task={rc.get('task')} proj_dim={rc.get('proj_dim')}")

    # Stash the trained logistic-link scalars a > 0 and b for plotting
    extra = {}
    try:
        import torch, os, glob
        sd_path = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
        if not os.path.exists(sd_path):
            cands = glob.glob(os.path.join(args.checkpoint_dir, "*.bin"))
            sd_path = cands[0] if cands else None
        if sd_path:
            sd = torch.load(sd_path, map_location="cpu")
            if "dist_a" in sd and "dist_b" in sd:
                a_raw = float(sd["dist_a"].reshape(-1)[0])
                b = float(sd["dist_b"].reshape(-1)[0])
                a = float(torch.nn.functional.softplus(torch.tensor(a_raw)))
                extra = {"a_raw": np.float32(a_raw), "a": np.float32(a), "b": np.float32(b)}
                print(f"[dump] head scalars: a_raw={a_raw:.4g} -> a={a:.4g} | b={b:.4g} "
                      f"| decision radius -b/a={-b/a:.4g}")
    except Exception as e:
        print(f"[dump] could not read a/b from checkpoint ({e}); plot will need --a_raw/--b")

    np.savez_compressed(
        args.out,
        labels=labels.astype(np.int8),
        ell=logits,
        trunk_contrast=trunk_contrast,
        head_contrast=head_contrast,
        trunk_norm=np.linalg.norm(trunk_contrast, axis=1).astype(np.float32),
        head_norm=np.linalg.norm(head_contrast, axis=1).astype(np.float32),
        total_reads=total_reads,
        chrom=chrom,
        anchor=anchor,
        checkpoint_dir=np.array(args.checkpoint_dir),
        data_csv=np.array(args.data_csv),
        **extra,
    )
    print(f"[write] {args.out}")

if __name__ == "__main__":
    main()
