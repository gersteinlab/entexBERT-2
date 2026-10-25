#!/usr/bin/env python
"""
entexbert2.analysis.ism_saliency -- calculates ISM saliency from any entexBERT-2 checkpoint

* regression  (Stage-1 binding trunk):  single-seq forward, target = binding score mu
    delta[w, i, b] = mu(sequence window w with position i mutated to base b) - mu(reference sequence window w)
    => importance[w, i] = -mean_{b != ref nucleotide} delta[w, i, b]
    * negative sign since mutating away from ref nucleotide hurts binding 
        (so delta[w, i, b] < 0 but should have importance[w, i] > 0) *

* classification (Stage-2 ASB head): twin forward
    ell = a*||z1 - z2|| + b (baseline ell(ref, ref) is minimum contrast)
    delta[w,i,b] = ell(mut,ref) - ell(ref,ref)
    importance[w, i] = +mean_{b != ref nucleotide} delta[w, i, b]
    * positive sign since mutating away from ref nucleotide is likely to create more contrast 
        (so delta[w, i, b] > 0, which matches importance[w, i] > 0 goal) *

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""
import argparse, numpy as np, pandas as pd

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}

# For Stage-1 checkpoints
def build_mutant_list(seqs):
    """Flatten [ref, all single mutants] with an index of (window, pos, base_idx); baseline: pos = -1"""
    inputs, index = [], []
    for w, s in enumerate(seqs):
        inputs.append(s); index.append((w, -1, -1))
        for i, ch in enumerate(s):
            for b in BASES:
                if b == ch:
                    continue
                inputs.append(s[:i] + b + s[i + 1:])
                index.append((w, i, B2I[b]))
    return inputs, index

# For Stage-2 checkpoints
def build_mutant_pairs(seqs, partners=None):
    """
    Twin-ISM inputs: mutate window1, hold window2 fixed
    - baseline (pos = -1) is [seq, partner]
    - same (window, pos, base_idx) indexing as above

    partners=None -> window2 = a copy of the reference window (baseline ell(ref,ref) = minimum contrast)
    partners=list -> window2 = the actual paired alt window for that locus (baseline is the real observed allelic contrast)"""

    inputs, index = [], []
    for w, s in enumerate(seqs):
        p = s if partners is None else partners[w]
        inputs.append([s, p]); index.append((w, -1, -1)) # baseline: ell(seq, partner)
        for i, ch in enumerate(s):
            for b in BASES:
                if b == ch:
                    continue
                inputs.append([s[:i] + b + s[i + 1:], p]) # mutate w1, hold w2 = partner
                index.append((w, i, B2I[b]))
    return inputs, index

def assemble(seqs, index, scores, task):
    """Flat scores -> per-window onehot / delta / importance / hyp / contribution arrays"""
    N = len(seqs); L = len(seqs[0])
    base = np.full(N, np.nan, dtype=np.float64)
    delta = np.zeros((N, L, 4), dtype=np.float32)
    onehot = np.zeros((N, L, 4), dtype=np.float32)
    for w, s in enumerate(seqs):
        for i, ch in enumerate(s):
            if ch in B2I:
                onehot[w, i, B2I[ch]] = 1.0
    for (w, i, b), sc in zip(index, scores):
        if i == -1:
            base[w] = sc
    for (w, i, b), sc in zip(index, scores):
        if i == -1:
            continue
        delta[w, i, b] = sc - base[w]
    # importance = mean over the 3 non-ref bases (ref entry is 0, so sum/3), sign by task
    sign = -1.0 if task == "regression" else +1.0
    importance = (sign * delta.sum(axis=2) / 3.0).astype(np.float32)
    # MoDISco hypothetical contributions: mean-center delta per position (ref base carries -mean)
    hyp = (delta - delta.mean(axis=2, keepdims=True)).astype(np.float32)
    contrib = (onehot * hyp).astype(np.float32)
    return dict(onehot=onehot, delta=delta, importance=importance,
                hyp_scores=hyp, contrib=contrib, base_score=base.astype(np.float32))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True,
                    help="any entexBERT-2 checkpoint; task auto-detected from run_config.json "
                         "(regression -> binding-trunk ISM, classification -> twin/head ISM)")
    ap.add_argument("--windows_csv", required=True, help="CSV with a sequence column")
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--rank_col", default="binding_label_raw",
                    help="rank windows descending by this column and take top n_windows (skip if absent)")
    ap.add_argument("--feature_col", default="feature_type",
                    help="keep only rows whose value contains this substring (skip if absent)")
    ap.add_argument("--feature_keep", default="peak",
                    help="value the feature_col must match (substring by default; e.g. 'peak')")
    ap.add_argument("--feature_exact", action="store_true",
                    help="require feature_col == feature_keep EXACTLY (use for as_label so "
                         "'AS' does not also match 'nonAS')")
    ap.add_argument("--n_windows", type=int, default=30)
    ap.add_argument("--twin_baseline", default="ref", choices=["ref", "alt"],
                    help="classification only. 'ref': hold window2 = copy of the reference window "
                         "(min-contrast baseline, symmetric motif map). 'alt': hold window2 = the "
                         "actual paired alt window from --partner_col (anchored at the real "
                         "observed allelic contrast).")
    ap.add_argument("--partner_col", default="sequence2",
                    help="column holding the alt/hap2 window; used only when --twin_baseline alt")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from entexbert2.model_io import run_inference, load_run_config

    task = load_run_config(a.checkpoint_dir).get("task", "regression")
    print(f"[ISM] checkpoint task = {task}  ->  "
          f"{'binding-trunk (single-seq mu)' if task == 'regression' else 'twin/head (ASB logit ell)'} ISM")

    df = pd.read_csv(a.windows_csv)
    if a.feature_col in df.columns:
        if a.feature_exact:
            df = df[df[a.feature_col].astype(str).str.strip() == a.feature_keep]
        else:
            df = df[df[a.feature_col].astype(str).str.contains(a.feature_keep, case=False, na=False)]
    if a.rank_col in df.columns:
        df = df.sort_values(a.rank_col, ascending=False)

    # n_windows <= 0 -> keep ALL selected rows
    df = (df if a.n_windows <= 0 else df.head(a.n_windows)).reset_index(drop=True)
    seqs = df[a.seq_col].astype(str).str.upper().str.strip().tolist()

    # windows are assumed ACGT-only (ensured by build_asb_ism_windows)
    if not seqs:
        raise SystemExit("no windows selected -- check --seq_col/--feature_col/--rank_col")
    L = len(seqs[0])
    bad = [i for i, s in enumerate(seqs) if len(s) != L]
    if bad:
        raise SystemExit(f"windows must be equal length for a stacked ISM array; {len(bad)} differ (L0={L})")

    if task == "classification":
        partners = None
        if a.twin_baseline == "alt":
            if a.partner_col not in df.columns:
                raise SystemExit(f"--twin_baseline alt needs '{a.partner_col}' in the CSV; "
                                 f"columns are {list(df.columns)}")
            partners = df[a.partner_col].astype(str).str.upper().str.strip().tolist()
            if any(len(p) != L for p in partners):
                raise SystemExit("partner windows must match window1 length L for stacked ISM")
        inputs, index = build_mutant_pairs(seqs, partners)
        kind = f"twin pairs [mut_w1, {a.twin_baseline}_w2]"
    else:
        inputs, index = build_mutant_list(seqs)
        kind = "single-seq"
    print(f"[ISM] {len(seqs)} windows x L={L}  -> {len(inputs)} forward inputs ({kind}; "
          f"{len(inputs)-len(seqs)} mutants + {len(seqs)} baselines)")

    # run_inference preserves input order and returns one score per input
    logits, _emb, rc = run_inference(a.checkpoint_dir, inputs, batch_size=a.batch_size, device=a.device)
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    if scores.shape[0] != len(inputs):
        raise SystemExit(f"scored {scores.shape[0]} != {len(inputs)} inputs "
                         f"(multi-track trunk? classification needs num_labels=1 contrast logit)")

    # pair index[k] with scores[k] positionally
    arr = assemble(seqs, index, scores, task)
    tb = a.twin_baseline if task == "classification" else "n/a"

    # carry per-window metadata (coords + depth + label) in seqs order
    meta = {c: df[c].to_numpy() for c in ("chr", "anchor", "total_reads", "as_label") if c in df.columns}
    np.savez_compressed(a.out, seqs=np.array(seqs), task=task, twin_baseline=tb, **arr, **meta)
    imp = arr["importance"]
    print(f"[ISM] saved -> {a.out} | task={task} | importance range [{imp.min():.3f},{imp.max():.3f}] "
          f"mean per-window peak {imp.max(axis=1).mean():.3f}")

if __name__ == "__main__":
    main()
