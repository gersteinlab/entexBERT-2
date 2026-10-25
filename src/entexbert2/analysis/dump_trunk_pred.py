#!/usr/bin/env python
"""
entexbert2.analysis.dump_trunk_pred
-- dump Stage-1 binding trunk predictions (mu) vs actual label on the held-out test split per cell

Currently configured for SCALAR TRUNKS ONLY (run_config num_labels == 1)

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""
import argparse
import numpy as np, pandas as pd

def main():
    from entexbert2.model_io import run_inference # keeps --help torch-free

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="the Stage-1 regression trunk")
    ap.add_argument("--test_csv", required=True, help="stage1_data/test.csv (just has sequence + label)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    df = pd.read_csv(a.test_csv)
    seqcol = "sequence" if "sequence" in df.columns else "sequence1"
    assert seqcol in df.columns and "label" in df.columns, df.columns.tolist()
    
    logits, _emb, rc = run_inference(a.checkpoint_dir, df[seqcol].astype(str).tolist(),
                                     a.batch_size, a.device, None)
    arr = np.asarray(logits)
    n_out = arr.shape[1] if arr.ndim > 1 else 1

    if int(rc.get("num_labels", 1)) > 1 or n_out > 1:
        raise SystemExit(
            f"FATAL: {a.checkpoint_dir} is a MULTI-TRACK trunk "
            f"(run_config num_labels={rc.get('num_labels')}, {n_out} output columns). This script "
            f"dumps a single scalar mu per locus and has no multi-track handling. Use a scalar "
            f"(num_labels=1) trunk, or extend this script and fig_binding_pred_grid")

    mu = arr.reshape(-1)
    out = pd.DataFrame({"label": df["label"].astype(float).to_numpy(), "mu": mu})
    out.to_csv(a.out, index=False)
    y, m = out["label"].to_numpy(), out["mu"].to_numpy()

    ok = np.isfinite(y) & np.isfinite(m)

    if ok.sum() > 1:
        yy, mm = y[ok], m[ok]
        r2 = 1.0 - np.sum((yy - mm) ** 2) / max(np.sum((yy - yy.mean()) ** 2), 1e-12)
    else:
        r2 = float("nan")

    drop = len(out) - int(ok.sum())

    print(f"[trunk_pred] {len(out)} loci -> {a.out}  r2={r2:.3f} (on {int(ok.sum())} finite"
          + (f", {drop} dropped" if drop else "") + f")  mu[{mu.min():.2f},{mu.max():.2f}]")

if __name__ == "__main__":
    main()
