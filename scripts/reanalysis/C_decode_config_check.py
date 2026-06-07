"""Task C — decode-config check.

Confirm from the stored configs that every under-test condition was decoded
greedy (T=0, top_p=1.0, top_k=1) at single decode. Produce one line so the
manuscript can state "all evaluation greedy, single decode" without
hand-waving. No generation; reads the YAML configs in configs/ only.

Configs audited (one per under-test condition):
  configs/baseline_7b_gpqa.yaml
  configs/r1_7b_baseline_gpqa.yaml
  configs/r1_7b_w2sr_gpqa.yaml
  configs/r1_7b_strong_gpqa.yaml
  configs/w2sr_student_gpqa.yaml
  configs/control_student_gpqa.yaml

Outputs:
  results/reanalysis/C_decode_config_check.md
  results/reanalysis/C_decode_config_check.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import REPO

OUT_MD   = REPO / "results/reanalysis/C_decode_config_check.md"
OUT_JSON = REPO / "results/reanalysis/C_decode_config_check.json"

CONFIGS = [
    ("baseline R1-7B",                 "configs/r1_7b_baseline_gpqa.yaml"),
    ("W2SR weak (R1-1.5B teacher)",    "configs/r1_7b_w2sr_gpqa.yaml"),
    ("W2SR strong (R1-14B teacher)",   "configs/r1_7b_strong_gpqa.yaml"),
    ("instruct baseline",              "configs/baseline_7b_gpqa.yaml"),
    ("instruct W2SR weak",             "configs/w2sr_student_gpqa.yaml"),
    ("instruct W2SR strong (control)", "configs/control_student_gpqa.yaml"),
]


# avoid pulling in pyyaml: configs are flat enough for regex
_KV = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*:\s*([^#\n]+?)\s*(?:#.*)?$", re.M)


def extract_decoding(text: str) -> dict[str, str]:
    """Pull temperature / top_p / top_k / n / epochs / sampling-style keys."""
    keys = {"temperature", "top_p", "top_k", "n", "epochs", "best_of"}
    out: dict[str, str] = {}
    for m in _KV.finditer(text):
        k = m.group(1); v = m.group(2).strip()
        if k in keys and k not in out:    # first occurrence is the under-test config
            out[k] = v
    return out


def is_greedy_single(d: dict[str, str]) -> tuple[bool, list[str]]:
    """Decision: greedy & single decode if T==0, top_p==1, top_k==1, and any
    n/best_of/epochs ≤ 1. Returns (ok, reasons-for-non-ok)."""
    fails = []
    t = d.get("temperature");  tp = d.get("top_p")
    tk = d.get("top_k");       n = d.get("n");       bo = d.get("best_of")
    ep = d.get("epochs")
    if t is None or float(t) != 0.0:                 fails.append(f"temperature={t!r}")
    if tp is not None and float(tp) != 1.0:          fails.append(f"top_p={tp!r}")
    if tk is not None and float(tk) != 1.0:          fails.append(f"top_k={tk!r}")
    if n is not None and float(n) > 1:               fails.append(f"n={n!r}")
    if bo is not None and float(bo) > 1:             fails.append(f"best_of={bo!r}")
    if ep is not None and float(ep) > 1:             fails.append(f"epochs={ep!r}")
    return (not fails), fails


def main():
    print("=" * 70)
    print("TASK C — decode-config check")
    print("=" * 70)
    rows = []
    all_ok = True
    for label, rel in CONFIGS:
        path = REPO / rel
        if not path.exists():
            print(f"  MISSING: {rel}")
            rows.append({"label": label, "config": rel, "missing": True}); all_ok = False
            continue
        text = path.read_text()
        d = extract_decoding(text)
        ok, fails = is_greedy_single(d)
        print(f"  {label:35s} {rel:42s} "
              f"T={d.get('temperature','?')}, top_p={d.get('top_p','?')}, "
              f"top_k={d.get('top_k','?')}, n={d.get('n','—')}, best_of={d.get('best_of','—')} "
              f"-> {'OK' if ok else 'FAIL ' + ','.join(fails)}")
        rows.append({"label": label, "config": rel, "decode": d,
                     "greedy_single": ok, "fails": fails})
        all_ok &= ok

    verdict = ("All under-test conditions decoded greedy, single decode (T=0, top_p=1.0, "
               "top_k=1; n/best_of unset or ≤1; epochs=1)." if all_ok else
               "NOT all conditions match greedy-single. See per-row fails.")
    print(f"\n{verdict}")

    OUT_JSON.write_text(json.dumps({"rows": rows,
                                     "all_greedy_single": all_ok,
                                     "verdict_line": verdict}, indent=2))
    md = ["# Task C — decode-config check\n",
          f"**Verdict:** {verdict}\n",
          "| condition | config path | T | top_p | top_k | n | best_of | greedy-single? |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("missing"):
            md.append(f"| {r['label']} | {r['config']} | — | — | — | — | — | MISSING |")
            continue
        d = r["decode"]
        md.append(f"| {r['label']} | {r['config']} | {d.get('temperature','—')} | "
                  f"{d.get('top_p','—')} | {d.get('top_k','—')} | {d.get('n','—')} | "
                  f"{d.get('best_of','—')} | "
                  f"{'OK' if r['greedy_single'] else 'FAIL: ' + ','.join(r['fails'])} |")
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_MD.relative_to(REPO)}  and  {OUT_JSON.relative_to(REPO)}")
    if not all_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
