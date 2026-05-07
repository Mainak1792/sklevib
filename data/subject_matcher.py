"""
data/subject_matcher.py
=======================
Builds a mapping from skeleton-folder subject names to vibration-folder
subject names, optionally using a subject metadata spreadsheet.

The matcher is intentionally dataset-agnostic: it uses normalised-string
and token-set matching so that minor spelling differences between the two
modality folder trees resolve automatically.  A ``folder_aliases`` dict
in ``Config`` handles cases where the same condition is stored under
different names in the two trees (e.g. ``oral`` vs ``cog``).
"""

import re
from pathlib import Path
from typing import Dict

import pandas as pd

from config import Config


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tokens(name: str):
    s = str(name).lower().strip()
    s = re.sub(r"_?\(\d+\)\s*$", "", s)
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def _norm(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"_?\(\d+\)\s*$", "", s)
    return re.sub(r"[\s_\-\(\)\.]+", "", s)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_subject_map(
    cfg: Config,
    extra_aliases: Dict[str, str] | None = None,
    xlsx_absent: set | None = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """Return ``{skel_subject_name: vib_folder_name}`` (empty string if unmatched).

    Parameters
    ----------
    cfg:
        Experiment config; uses ``skeleton_root``, ``vib_root``,
        ``subject_xlsx_path``, ``subject_xlsx_name_col``,
        ``subject_xlsx_prefix``.
    extra_aliases:
        Optional ``{skel_name: vib_name}`` overrides for subjects whose
        folder names differ between the two modality trees but are not
        resolved by the automatic normalisation.  Pass dataset-specific
        aliases here rather than hard-coding them in this module.
    xlsx_absent:
        Set of skeleton subject names that are intentionally absent from
        the spreadsheet (e.g. subjects recruited after the metadata was
        locked).  These are silently skipped during xlsx lookup.
    verbose:
        Print a coverage summary.
    """
    skel_root = Path(cfg.skeleton_root)
    vib_root = Path(cfg.vib_root)
    extra_aliases = extra_aliases or {}
    xlsx_absent = xlsx_absent or set()

    # ---- enumerate subjects in both trees ----
    skel_subjects = sorted({
        sd.name
        for v in skel_root.iterdir() if v.is_dir()
        for c in v.iterdir() if c.is_dir()
        for sd in c.iterdir() if sd.is_dir()
    })
    vib_subjects = sorted({
        sd.name
        for s in vib_root.iterdir() if s.is_dir()
        for c in s.iterdir() if c.is_dir()
        for sd in c.iterdir() if sd.is_dir()
    })
    vib_by_lower = {v.lower(): v for v in vib_subjects}
    vib_by_num: Dict[int, str] = {}
    for v in vib_subjects:
        nums = re.findall(r"\d+", v)
        if nums:
            vib_by_num[int(nums[0])] = v

    # ---- load xlsx index ----
    xlsx_by_norm: Dict[str, list] = {}
    xlsx_by_tok: Dict[tuple, list] = {}
    if cfg.subject_xlsx_path and Path(cfg.subject_xlsx_path).exists():
        try:
            df = pd.read_excel(cfg.subject_xlsx_path)
            col = (
                cfg.subject_xlsx_name_col
                if cfg.subject_xlsx_name_col in df.columns
                else df.columns[0]
            )
            for i, raw in enumerate(df[col].astype(str).tolist()):
                pid = f"{cfg.subject_xlsx_prefix}{i + 1}"
                entry = (i + 1, pid, raw)
                xlsx_by_norm.setdefault(_norm(raw), []).append(entry)
                xlsx_by_tok.setdefault(tuple(sorted(_tokens(raw))), []).append(entry)
            if verbose:
                print(f"[matcher] xlsx rows loaded: {len(df)}")
        except Exception as e:
            print(f"[matcher] xlsx load failed: {e}")

    def _xlsx_to_vib(pid: str) -> str | None:
        if pid.lower() in vib_by_lower:
            return vib_by_lower[pid.lower()]
        nums = re.findall(r"\d+", pid)
        if nums and int(nums[0]) in vib_by_num:
            return vib_by_num[int(nums[0])]
        return None

    def _pick(entries, skel: str):
        if len(entries) == 1:
            return entries[0]
        m = re.search(r"\((\d+)\)\s*$", skel)
        k = int(m.group(1)) if m else None
        return entries[min(k, len(entries) - 1)] if k is not None else entries[0]

    def _resolve(skel: str) -> str | None:
        if skel.lower() in {s.lower() for s in xlsx_absent}:
            return None
        # extra_aliases override
        if skel.lower() in {k.lower(): v for k, v in extra_aliases.items()}:
            cand = extra_aliases[skel]
            e = xlsx_by_norm.get(_norm(cand)) or xlsx_by_tok.get(
                tuple(sorted(_tokens(cand)))
            )
            if e:
                r = _xlsx_to_vib(_pick(e, skel)[1])
                if r:
                    return r
        # normalised-string match
        n = _norm(skel)
        if n in xlsx_by_norm:
            r = _xlsx_to_vib(_pick(xlsx_by_norm[n], skel)[1])
            if r:
                return r
        # token-set match
        t = tuple(sorted(_tokens(skel)))
        if t in xlsx_by_tok:
            r = _xlsx_to_vib(_pick(xlsx_by_tok[t], skel)[1])
            if r:
                return r
        # trailing-number heuristic
        if re.search(r"[_\-]\d+\s*$", skel):
            nums = re.findall(r"\d+", skel)
            if nums and int(nums[-1]) in vib_by_num:
                return vib_by_num[int(nums[-1])]
        # exact lower match in vib tree
        if skel.lower() in vib_by_lower:
            return vib_by_lower[skel.lower()]
        return None

    mapping = {s: _resolve(s) or "" for s in skel_subjects}
    n_mapped = sum(1 for v in mapping.values() if v)
    if verbose:
        pct = 100 * n_mapped / max(1, len(mapping))
        print(
            f"[matcher] mapped {n_mapped}/{len(mapping)} ({pct:.1f}%)"
        )
        unm = [k for k, v in mapping.items() if not v]
        if unm:
            print(
                f"[matcher] UNMAPPED ({len(unm)}): {unm[:10]}"
                + ("..." if len(unm) > 10 else "")
            )
            print(
                "[matcher] unmapped subjects will be skeleton-only at train time."
            )
    return mapping
