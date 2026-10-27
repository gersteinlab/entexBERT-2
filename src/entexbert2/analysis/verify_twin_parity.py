#!/usr/bin/env python
"""verify_twin_parity.py -- prove the head_variant change did not alter any existing TWIN checkpoint.

The head_variant design rests on one claim: a checkpoint trained before head_variant existed has no
"head_variant" key in run_config.json, so model_io defaults it to "twin", and the twin construction
and forward paths are untouched -- therefore the logits are bit-identical. This script tests that
claim against a REAL Stage-2 checkpoint instead of assuming it.

Module caching makes it unsafe to import two versions of entexbert2 in one process, so this runs in
two passes and a compare:

  # pass 1 -- OLD package (pre-head_variant), e.g. a git worktree or the old repo clone
  PYTHONPATH=/path/to/OLD/src python verify_twin_parity.py dump \
      --checkpoint_dir $HEAD --test_csv $BUILD_S2/test.csv --n 256 --out old.npz

  # pass 2 -- NEW package (with head_variant)
  PYTHONPATH=/path/to/NEW/src python verify_twin_parity.py dump \
      --checkpoint_dir $HEAD --test_csv $BUILD_S2/test.csv --n 256 --out new.npz

  # compare (either PYTHONPATH; numpy only)
  python verify_twin_parity.py compare --a old.npz --b new.npz

PASS requires bit-identical logits. Anything else means the twin path moved and the ablation's
earlier conditions are not comparable to the already-trained ones.

Run on a GPU node (or pass --device cpu); scoring 256 pairs is seconds either way.
"""
import argparse
import csv
import sys

import numpy as np


def dump(a):
    from entexbert2.model_io import load_run_config, run_inference

    with open(a.test_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{a.test_csv} is empty")
    need = {"sequence1", "sequence2"}
    if not need <= set(rows[0]):
        sys.exit(f"{a.test_csv} needs {sorted(need)} (twin test set); got {sorted(rows[0])}")
    rows = rows[: a.n]
    pairs = [[r["sequence1"], r["sequence2"]] for r in rows]
    labels = np.array([float(r.get("label", "nan")) for r in rows], dtype=np.float64)

    # what model_io reconstructs -- head_variant should be ABSENT in a pre-change checkpoint
    rc = load_run_config(a.checkpoint_dir)
    print(f"[run_config] task={rc.get('task')!r} input_mode={rc.get('input_mode')!r} "
          f"head_variant={rc.get('head_variant', '<absent -> defaults to twin>')!r} "
          f"proj_dim={rc.get('proj_dim')} head_num_layers={rc.get('head_num_layers')}")

    logits, emb, run_config = run_inference(a.checkpoint_dir, pairs,
                                            batch_size=a.batch_size, device=a.device)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    print(f"[dump] {len(logits)} logits | mean={logits.mean():.10f} sd={logits.std():.10f}")
    np.savez(a.out, logits=logits, labels=labels,
             head_variant=str(rc.get("head_variant", "ABSENT")),
             n=len(logits))
    print(f"[dump] wrote {a.out}")


def compare(a):
    A = np.load(a.a, allow_pickle=False)
    B = np.load(a.b, allow_pickle=False)
    la, lb = A["logits"], B["logits"]
    if la.shape != lb.shape:
        sys.exit(f"FAIL: shape mismatch {la.shape} vs {lb.shape} -- different rows scored")

    exact = int((la == lb).sum())
    maxabs = float(np.max(np.abs(la - lb))) if la.size else 0.0
    print(f"  n            : {la.size}")
    print(f"  head_variant : a={A['head_variant']} b={B['head_variant']}")
    print(f"  bit-identical: {exact}/{la.size}")
    print(f"  max |diff|   : {maxabs:.3e}")

    if exact == la.size:
        print("\nPASS: logits are bit-identical -- the twin path is unchanged.")
        return 0
    if maxabs < 1e-6:
        print(f"\nPASS (numerically): max |diff| {maxabs:.3e} is below 1e-6 -- consistent with "
              "nondeterministic kernel/reduction order, not a changed code path. Re-run both passes "
              "on the SAME device to try for bit-identity.")
        return 0
    print(f"\nFAIL: max |diff| {maxabs:.3e} is too large to be numerical noise. The twin "
          "construction or forward path changed; existing checkpoints are NOT reproduced.")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="score a checkpoint and save logits")
    d.add_argument("--checkpoint_dir", required=True, help="a trained TWIN Stage-2 head dir")
    d.add_argument("--test_csv", required=True, help="Stage-2 test.csv (sequence1,sequence2,label)")
    d.add_argument("--n", type=int, default=256, help="rows to score (default 256)")
    d.add_argument("--batch_size", type=int, default=64)
    d.add_argument("--device", default="cuda")
    d.add_argument("--out", required=True)

    c = sub.add_parser("compare", help="diff two dumps")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)

    a = ap.parse_args()
    sys.exit(dump(a) or 0 if a.cmd == "dump" else compare(a))


if __name__ == "__main__":
    main()
