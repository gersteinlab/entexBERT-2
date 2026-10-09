#!/usr/bin/env python

"""
personal_seq.py -- lift hg38 reference loci into an individual's phased haplotype genomes and extract personal windows

Two vcf2diploid liftOver chains per individual map the reference to each haplotype:
    ENC00X_maternal.chain   REF -> maternal    (hap1)
    ENC00X_paternal.chain   REF -> paternal    (hap2)

For a het SNV at REF (chrom, pos0) [0-based] with alleles a1 (hap1_allele), a2 (hap2_allele):

  1. Lift pos0 through each chain -> maternal coord / paternal coord (None if deleted there)

  2. Extract a FIXED-length window (+/- flank bp) from each haplotype FASTA,
        re-centered on the lifted variant -> both windows are (left+right+1) bp with the variant at offset `left`
        ** so indels within the window shift how much ref span it covers but not its length **

  3. Read the focal base in each personal window; ASSIGN maternal/paternal -> hap1/hap2 by matching the focal bases to (a1, a2)

UCSC chain format (0-based, half-open; target = REF, query = haplotype):
    chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
    size dt dq        # aligned block `size`, then target-gap dt and query-gap dq before next block
    ...
    size              # final block (no trailing gaps)
    <blank line>
vcf2diploid emits +/+ strands (asserted in this file to be safe)

NOTE: lift is only valid if the hetSNV calls and vcf2diploid used the SAME reference windows (hg38)

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""

import bisect
from collections import defaultdict

# ---------------------------------------------------------------------------------------------
# Chain parsing + point lift
# ---------------------------------------------------------------------------------------------
class _Chain:
    """One `chain` record: aligned blocks mapping target [tStart,tEnd) -> query [qStart,qEnd)"""
    __slots__ = ("t_name", "t_size", "t_start", "t_end", "q_name", "q_size",
                 "q_strand", "q_start", "q_end", "blocks",
                 "_tstarts", "_qstarts", "_sizes")

    def __init__(self, header_fields):
        # header_fields = tokens after the 'chain' keyword
        (_score, self.t_name, self.t_size, t_strand, self.t_start, self.t_end,
         self.q_name, self.q_size, self.q_strand, self.q_start, self.q_end, *_id) = header_fields

        self.t_size = int(self.t_size); self.t_start = int(self.t_start); self.t_end = int(self.t_end)
        self.q_size = int(self.q_size); self.q_start = int(self.q_start); self.q_end = int(self.q_end)

        if t_strand != "+":
            raise ValueError(f"chain {self.t_name}->{self.q_name}: target strand {t_strand!r} != '+'")

        if self.q_strand != "+":
            raise ValueError(
                f"chain {self.t_name}->{self.q_name}: query strand {self.q_strand!r} != '+'. "
                "extract_window reads the forward strand and does not reverse-complement, so a '-' "
                "query would deliver complemented bases without error. "
                "Please re-generate the chain with a '+' query!")
        
        self.blocks = [] # (size, dt, dq)
        self._tstarts = None # per-block index, built on first lift()

    def _index_blocks(self):
        """Precompute each block's target/query start so lift() can binary-search"""
        t, q = self.t_start, self.q_start
        tstarts, qstarts, sizes = [], [], []
        for (size, dt, dq) in self.blocks:
            tstarts.append(t)
            qstarts.append(q)
            sizes.append(size)
            t += size + dt
            q += size + dq
        self._tstarts, self._qstarts, self._sizes = tstarts, qstarts, sizes

    def lift(self, tpos):
        """Lift a 0-based TARGET position to a 0-based QUERY position,
        or None if it falls in a target-only gap (i.e. the base is deleted in this haplotype)"""
        if tpos < self.t_start or tpos >= self.t_end:
            return None
        if self._tstarts is None:
            self._index_blocks()
        i = bisect.bisect_right(self._tstarts, tpos) - 1 # last block starting at or before tpos
        if i < 0:
            return None
        if tpos < self._tstarts[i] + self._sizes[i]: # inside the aligned block -> 1:1
            return self._qstarts[i] + (tpos - self._tstarts[i])
        return None


def parse_chain_file(path):
    """Parse a (possibly multi-chromosome) UCSC chain file -> {target_chrom: [_Chain, ...]}"""
    index = defaultdict(list)
    cur = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                cur = None
                continue
            if line.startswith("chain"):
                cur = _Chain(line.split()[1:])
                index[cur.t_name].append(cur)
            elif cur is not None:
                parts = line.split()
                if len(parts) == 1:
                    cur.blocks.append((int(parts[0]), 0, 0)) # final block
                else:
                    cur.blocks.append((int(parts[0]), int(parts[1]), int(parts[2])))

    # To be safe, check that the chain's blocks exactly span its declared target/query intervals
    for chains in index.values():
        for ch in chains:
            t_span = sum(size + dt for size, dt, _ in ch.blocks)
            q_span = sum(size + dq for size, _, dq in ch.blocks)
            if t_span != ch.t_end - ch.t_start or q_span != ch.q_end - ch.q_start:
                raise ValueError(
                    f"chain {ch.t_name}->{ch.q_name}: blocks span t={t_span}/q={q_span} but header "
                    f"declares t={ch.t_end - ch.t_start}/q={ch.q_end - ch.q_start} -- truncated or "
                    f"malformed chain file")

    return dict(index)

def lift_point(chain_index, chrom, pos0):
    """
    Lift 0-based (chrom, pos0) through the first covering chain -> (q_name, q_pos0) or None

    NOTE: if several chains cover `pos0`, the FIRST in file order wins;
    vcf2diploid emits one chain per chromosome so this does not arise in the entexBERT-2 implementation,
    but note that UCSC liftOver files would need to be score-sorted for this to pick the best alignment
    """
    for ch in chain_index.get(chrom, ()):
        q = ch.lift(pos0)
        if q is not None:
            return (ch.q_name, q)
    return None

# ---------------------------------------------------------------------------------------------
# Haplotype window extraction (FASTA-agnostic: anything supporting fa[chrom][a:b] -> str)
# ---------------------------------------------------------------------------------------------
_HAP_SUFFIXES = ("_maternal", "_paternal", "_hap1", "_hap2")

def _base_chrom(name):
    """Strip a haplotype/parent tag: 'chr1_maternal'->'chr1', 'chr1_hap1'->'chr1', 'chr1'->'chr1'"""
    for suf in _HAP_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name

def _resolve_chrom(fasta, q_name):
    """Map a chain query contig (e.g. 'chr1_maternal') to a haplotype-FASTA key"""
    keys = list(getattr(fasta, "keys", lambda: [])())
    kset = set(keys)
    if q_name in kset:                                  # exact match (same naming)
        return q_name
    base = _base_chrom(q_name)                          # chr1_maternal -> chr1
    if base in kset:                                    # FASTA uses the bare chrom
        return base
    hits = [k for k in keys if _base_chrom(k) == base]  # FASTA tags differently (chr1_maternal -> chr1_hap1)
    return hits[0] if len(hits) == 1 else None

def extract_window(fasta, chrom, q_pos0, left, right):
    """
    Fixed-length window [q_pos0-left, q_pos0+right] (length left+right+1), variant at offset `left`,
    NOTE: `chrom` must ALREADY be a FASTA key (resolve with _resolve_chrom first);
    Returns (seq_upper, offset), or None if the window runs off either end of the contig
    """
    start = q_pos0 - left
    end = q_pos0 + right + 1
    if start < 0:
        return None
    seq = str(fasta[chrom][start:end]).upper()
    if len(seq) != left + right + 1:                 # ran off the chromosome end
        return None
    return seq, left

# ---------------------------------------------------------------------------------------------
# PersonalGenome: bind (hap1 chain+FASTA, hap2 chain+FASTA) and build validated allele-matched pairs
# ---------------------------------------------------------------------------------------------
class PersonalGenome:
    """
    hap1 = (maternal chain, hap1 FASTA), hap2 = (paternal chain, hap2 FASTA) per the README default.
    pair_windows() lifts a REF het SNV into both haplotypes, extracts variant-centered windows, and
    assigns them to hap1/hap2 by ALLELE-MATCH (so a sex-chrom mat/pat swap is auto-corrected).
    """
    def __init__(self, hap1_chain_index, hap1_fasta, hap2_chain_index, hap2_fasta, donor=None,
                validate=True):
        self.h1_chain, self.h1_fa = hap1_chain_index, hap1_fasta
        self.h2_chain, self.h2_fa = hap2_chain_index, hap2_fasta
        self.donor = donor
        self.stats = defaultdict(int)   # total / ok / swap / mismatch / hapN_deleted / hapN_unresolved_contig / hapN_off_contig_end
        self._chrom_cache = {}          # (hap tag, chain query name) -> FASTA key or None
        if validate:
            self._validate_chain_fasta()

    def _validate_chain_fasta(self):
        """Fail fast when a chain was built against a DIFFERENT FASTA than the one supplied: for
        every chain whose query contig resolves, the chain's qSize must equal that contig's length.
        Otherwise the mismatch surfaces only as a mysteriously low allele-match rate after a full
        run. Silently skipped if the FASTA object does not support len() (this class is otherwise
        FASTA-agnostic: it only needs fa[chrom][a:b])."""
        who = f"donor {self.donor}: " if self.donor else ""
        for tag, chain_index, fasta in (("hap1", self.h1_chain, self.h1_fa),
                                        ("hap2", self.h2_chain, self.h2_fa)):
            bad, n_resolved, n_chains, unsupported = [], 0, 0, False
            for chains in chain_index.values():
                for ch in chains:
                    n_chains += 1
                    key = self._resolve_cached(tag, fasta, ch.q_name)
                    if key is None:
                        continue
                    try:
                        flen = len(fasta[key])
                    except TypeError:
                        unsupported = True
                        break
                    n_resolved += 1
                    if flen != ch.q_size:
                        bad.append(f"{ch.q_name}->{key}: chain qSize={ch.q_size} vs FASTA length={flen}")
                if unsupported:
                    break
            if unsupported or not n_chains:
                continue
            if bad:
                raise ValueError(
                    f"{who}{tag} chain/FASTA mismatch on {len(bad)}/{n_resolved} contig(s) -- the chain "
                    f"was built against a different haplotype FASTA: " + "; ".join(bad[:5]))
            if n_resolved == 0:
                raise ValueError(
                    f"{who}{tag}: none of the {n_chains} chain query contig(s) resolve to a key in the "
                    f"supplied FASTA (chain names e.g. {sorted({c.q_name for cs in chain_index.values() for c in cs})[:3]}; "
                    f"FASTA keys e.g. {list(fasta.keys())[:3]}) -- wrong FASTA, or a naming convention "
                    f"_resolve_chrom does not handle.")
            if n_resolved < n_chains:
                print(f"[personal] NOTE: {who}{tag}: {n_chains - n_resolved}/{n_chains} chain contigs "
                      f"are absent from the FASTA (loci there will count as {tag}_unresolved_contig).")

    def _resolve_cached(self, tag, fasta, q_name):
        key = (tag, q_name)
        if key not in self._chrom_cache:
            self._chrom_cache[key] = _resolve_chrom(fasta, q_name)
        return self._chrom_cache[key]

    def pair_windows(self, chrom, pos0, a1, a2, left=128, right=128):
        """
        chrom, pos0 (0-based): REF locus
        a1 = hap1_allele, a2 = hap2_allele (single bases)
        Returns dict(seq1, seq2, offset, swapped) on success, or (None, reason) on failure
        """
        self.stats["total"] += 1
        a1, a2 = str(a1).upper(), str(a2).upper()
        if a1 == a2:
            self.stats["not_het"] += 1
            return None, f"alleles are identical ({a1}) -- not a heterozygous SNV"

        # lift through BOTH chains (maternal / paternal), extract a variant-centered window from each
        def side(chain_index, fasta, tag):
            """-> (window, None) or (None, reason). Reasons are distinct so the stats tell a
            CONFIG error (contig unresolved -> every locus on that chrom fails) apart from
            biology (variant deleted in this haplotype) and geometry (window off contig end)."""
            lifted = lift_point(chain_index, chrom, pos0)
            if lifted is None:
                return None, f"{tag}_deleted" # target gap, or no covering chain
            q_name, q_pos0 = lifted
            key = self._resolve_cached(tag, fasta, q_name)
            if key is None:
                return None, f"{tag}_unresolved_contig" # chain query contig absent from the FASTA
            win = extract_window(fasta, key, q_pos0, left, right)
            if win is None:
                return None, f"{tag}_off_contig_end"
            return win, None

        m, m_why = side(self.h1_chain, self.h1_fa, "hap1") # maternal side (nominal hap1)
        p, p_why = side(self.h2_chain, self.h2_fa, "hap2") # paternal side (nominal hap2)
        if m is None or p is None:
            why = m_why or p_why # one increment per locus
            self.stats[why] += 1
            return None, f"window unavailable ({m_why or 'hap1 ok'}; {p_why or 'hap2 ok'})"

        (m_seq, off), (p_seq, _) = m, p
        m_base, p_base = m_seq[off], p_seq[off]

        # assign by allele-match; auto-correct a maternal/paternal <-> hap1/hap2 swap (sex chroms)
        if m_base == a1 and p_base == a2:
            self.stats["ok"] += 1
            return {"seq1": m_seq, "seq2": p_seq, "offset": off, "swapped": False}, None
        if m_base == a2 and p_base == a1:
            self.stats["ok"] += 1; self.stats["swap"] += 1
            return {"seq1": p_seq, "seq2": m_seq, "offset": off, "swapped": True}, None

        self.stats["mismatch"] += 1
        return None, (f"personal allele mismatch at {chrom}:{pos0} "
                      f"(maternal base {m_base!r}, paternal base {p_base!r}; expected {{{a1},{a2}}})")

    def match_rate(self):
        t = self.stats["total"]
        return (self.stats["ok"] / t) if t else float("nan")
