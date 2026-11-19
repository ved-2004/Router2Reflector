"""Stage 2 analysis: end-to-end accuracy of every (size, option, gate) on Validation and Test.

Per-row runs come from rag_eval/results/runs/qwen2.5-{size}/{split}/{option}.csv (pulled from the volume).
The answer for a row = the with-context prediction when the context is injected (option produced a
context AND the gate fires), otherwise the existing direct prediction (outputs/pipeline_runs/*, not
recomputed). R0 (rag_ft_filt from the existing runs) is included for reference.

Gates (all thresholds fitted on Validation per (size, option), then frozen and applied to Test):
  always    : inject whenever the option produced a context
  route     : never inject for AQUA-RAT or ReClor (no fitting)
  H         : route + inject only if direct_H >= tau (tau grid = Validation quantiles; ties -> higher tau)
  nli       : route + inject only if max_o support(o) >= tau and support gap (top1 - top2) >= delta
              support(o) = max over context passages of P(entail | passage, last question sentence + option)
  src       : per-source on/off chosen on Validation (on only if it strictly improved that source)
Stats: accuracy, delta vs direct (pp) with paired-bootstrap 95% CI (2000 resamples), fixed/broken counts,
per-source delta, injection rate.
Outputs: rag_eval/results/stage2_accuracy.csv, stage2_per_source.csv, stage2_rows/*.csv.gz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, ROOT, SIZES, load_direct  # noqa

SPLITS = ["validation", "test"]
OFF_SOURCES = {"AQUA-RAT", "ReClor"}
RNG = np.random.default_rng(0)
N_BOOT = 2000
NLI = json.loads((RES / "nli.json").read_text()) if (RES / "nli.json").exists() else {}


def load_option(size, split, option):
    d = load_direct(size, split)[["row_id", "source", "gold", "direct_pred", "direct_correct", "direct_H",
                                  "direct_margin", "rag_ft_filt_pred", "rag_ft_filt_correct"]]
    if option == "R0_ft_filt":
        d["ctx_pred"], d["ctx_correct"] = d["rag_ft_filt_pred"], d["rag_ft_filt_correct"]
        d["has_ctx"] = d["rag_ft_filt_pred"].notna()
    else:
        r = pd.read_csv(RES / "runs" / f"qwen2.5-{size}" / split / f"{option}.csv")
        r = r.rename(columns={"pred": "ctx_pred", "correct": "ctx_correct", "H": "ctx_H", "margin": "ctx_margin"})
        d = d.merge(r[["row_id", "has_ctx", "ctx_pred", "ctx_correct", "ctx_H", "ctx_margin"]], on="row_id")
    d["has_ctx"] = d["has_ctx"].fillna(False).astype(bool) & d["ctx_pred"].notna()
    sup = NLI.get(split, {}).get(option, {})
    s1, s2 = [], []
    for rid in d["row_id"]:
        o = sup.get(str(rid))
        if not o:
            s1.append(np.nan); s2.append(np.nan); continue
        v = sorted((max(x) for x in o.values()), reverse=True)
        s1.append(v[0]); s2.append(v[1] if len(v) > 1 else 0.0)
    d["sup1"], d["sup_gap"] = s1, np.array(s1) - np.array(s2)
    return d


def apply(d, mask):
    use = d["has_ctx"] & mask
    corr = np.where(use, d["ctx_correct"], d["direct_correct"]).astype(float)
    return corr, use


def stats(d, corr, use):
    dc = d["direct_correct"].to_numpy(float)
    diff = corr - dc
    n = len(d)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boots = diff[idx].mean(1)
    return {"n": n, "acc": round(corr.mean() * 100, 2), "acc_direct": round(dc.mean() * 100, 2),
            "delta_pp": round(diff.mean() * 100, 2), "ci_lo": round(np.percentile(boots, 2.5) * 100, 2),
            "ci_hi": round(np.percentile(boots, 97.5) * 100, 2), "fixed": int(((corr == 1) & (dc == 0)).sum()),
            "broken": int(((corr == 0) & (dc == 1)).sum()), "inject_rate": round(float(use.mean()) * 100, 1)}


def route_mask(d):
    return ~d["source"].isin(OFF_SOURCES)


def fit_H(v):
    best = (-1, None)
    for tau in sorted(set(np.round(np.quantile(v["direct_H"], np.linspace(0, 1, 21)), 6)), reverse=True):
        acc = apply(v, route_mask(v) & (v["direct_H"] >= tau))[0].mean()
        if acc > best[0] + 1e-12:
            best = (acc, tau)
    return best[1]


def fit_nli(v):
    best = (-1, None)
    if v["sup1"].isna().all():
        return None
    for tau in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        for gap in (0.6, 0.4, 0.2, 0.1, 0.0):
            acc = apply(v, route_mask(v) & (v["sup1"] >= tau) & (v["sup_gap"] >= gap))[0].mean()
            if acc > best[0] + 1e-12:
                best = (acc, (tau, gap))
    return best[1]


def fit_src(v):
    on = []
    for s, g in v.groupby("source"):
        corr, _ = apply(g, pd.Series(True, index=g.index))
        if corr.mean() > g["direct_correct"].mean() + 1e-12:
            on.append(s)
    return on


def gate_masks(v, t):
    """Fit on validation frame v, return {gate: (param, mask_fn)} to apply to test frame t (and v)."""
    out = {"always": (None, lambda d: pd.Series(True, index=d.index)), "route": (None, route_mask)}
    tau = fit_H(v)
    out["H"] = (tau, lambda d, tau=tau: route_mask(d) & (d["direct_H"] >= tau))
    p = fit_nli(v)
    if p is not None:
        out["nli"] = (p, lambda d, p=p: route_mask(d) & (d["sup1"] >= p[0]) & (d["sup_gap"] >= p[1]))
    on = fit_src(v)
    out["src"] = (on, lambda d, on=on: d["source"].isin(on))
    return out


def main(options):
    recs, src_recs = [], []
    (RES / "stage2_rows").mkdir(exist_ok=True)
    for size in SIZES:
        for option in ["R0_ft_filt"] + options:
            try:
                v, t = load_option(size, "validation", option), load_option(size, "test", option)
            except FileNotFoundError as e:
                print("missing", size, option, e); continue
            for gate, (param, fn) in gate_masks(v, t).items():
                for split, d in (("validation", v), ("test", t)):
                    corr, use = apply(d, fn(d))
                    rec = {"size": size, "option": option, "gate": gate, "split": split,
                           "param": json.dumps(param), **stats(d, corr, use)}
                    recs.append(rec)
                    for s, idx in d.groupby("source").groups.items():
                        sub = d.loc[idx]
                        src_recs.append({"size": size, "option": option, "gate": gate, "split": split, "source": s,
                                         "n": len(sub), "acc": round(corr[d.index.get_indexer(idx)].mean() * 100, 1),
                                         "acc_direct": round(sub["direct_correct"].mean() * 100, 1),
                                         "delta_pp": round((corr[d.index.get_indexer(idx)].mean() - sub["direct_correct"].mean()) * 100, 1),
                                         "fixed": int(((corr[d.index.get_indexer(idx)] == 1) & (sub["direct_correct"] == 0)).sum()),
                                         "broken": int(((corr[d.index.get_indexer(idx)] == 0) & (sub["direct_correct"] == 1)).sum())})
                    if gate == "always":
                        out = d.assign(final_correct=corr, injected=use)
                        out.to_csv(RES / "stage2_rows" / f"{size}_{option}_{split}.csv.gz", index=False)
    df = pd.DataFrame(recs)
    df.to_csv(RES / "stage2_accuracy.csv", index=False)
    pd.DataFrame(src_recs).to_csv(RES / "stage2_per_source.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 1000); pd.set_option("display.max_columns", 30)
    t = df[df.split == "test"]
    for gate in ("always", "route", "H", "nli", "src"):
        g = t[t.gate == gate]
        if g.empty:
            continue
        g = g.assign(cell=g.apply(lambda r: f"{r.delta_pp:+.1f} [{r.ci_lo:+.1f},{r.ci_hi:+.1f}]", axis=1))
        print(f"\n=== Test delta pp vs direct, gate={gate}")
        print(g.pivot(index="option", columns="size", values="cell")[SIZES])


if __name__ == "__main__":
    main(sys.argv[1:])
