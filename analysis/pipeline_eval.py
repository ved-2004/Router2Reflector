"""
Offline evaluation of combined Router x RAG x Reflector pipelines.

Inputs: outputs/pipeline_runs/qwen2.5-{size}/{split}.csv  (from modal_app/r2r_modal.py), which hold
every condition (direct, 4 reflections, RAG raw/filtered, evidence-reconstruction + context) on every row.
Optional: the original 3B/7B router (--router_dir) for the "Final Design" router flag.

Protocol: every threshold / prompt-style / classifier is fitted on Validation (or Train for the learned
selector) and reported on Test. Confidence regimes use Validation percentiles (fixes the test-set
percentile leakage in the Reflector notebook).

  python analysis/pipeline_eval.py --runs outputs/pipeline_runs --out outputs/pipeline_eval
"""

from __future__ import annotations

import argparse
import warnings
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reflector_core import PROMPT_STYLE_ORDER, assign_confidence_regime, compute_blind_thresholds  # noqa: E402

warnings.filterwarnings("ignore")
SIZES = ["3B", "7B", "14B", "32B"]
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def load(runs: Path, size: str, split: str) -> pd.DataFrame | None:
    p = runs / f"qwen2.5-{size}" / f"{split}.csv"
    return pd.read_csv(p) if p.exists() else None


def regime(df, cond, th):
    return np.array([assign_confidence_regime(h, m, th) if not (pd.isna(h) or pd.isna(m)) else "unknown"
                     for h, m in zip(df[f"{cond}_H"], df[f"{cond}_margin"])])


def col(df, cond, field, fallback="direct"):
    """Value of a condition's field, falling back to the direct pass where the condition did not run."""
    a = df[f"{cond}_{field}"] if f"{cond}_{field}" in df else pd.Series(np.nan, index=df.index)
    return a.where(a.notna(), df[f"{fallback}_{field}"]).values


HAS = {"raw": "has_raw", "filt": "has_filt", "ft_raw": "has_raw_ft", "ft_filt": "has_filt_ft"}


class Out:
    """Per-row pipeline result."""

    def __init__(self, n):
        self.pred = np.empty(n, dtype=object)
        self.margin = np.zeros(n)
        self.H = np.zeros(n)
        self.abstain = np.zeros(n, bool)
        self.calls = np.ones(n)        # LLM forward passes (direct pass always runs)
        self.retr = np.zeros(n)        # retrieval calls

    def set(self, mask, df, cond):
        self.pred[mask] = col(df, cond, "pred")[mask]
        self.margin[mask] = col(df, cond, "margin")[mask]
        self.H[mask] = col(df, cond, "H")[mask]


def metrics(o: Out, df: pd.DataFrame, base_correct: np.ndarray) -> dict:
    gold = df.gold.values
    forced = (o.pred == gold).astype(int)
    ans = ~o.abstain
    sel = forced[ans].mean() if ans.any() else np.nan
    # risk-coverage on final margin (higher margin = more confident)
    order = np.argsort(-o.margin, kind="stable")
    c = forced[order]
    aurc = float(np.mean(np.cumsum(1 - c) / np.arange(1, len(c) + 1)))
    k80 = int(0.8 * len(c))
    diff = forced - base_correct
    idx = RNG.integers(0, len(diff), (2000, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(1), [2.5, 97.5]) * 100
    return {"FA": forced.mean() * 100, "dFA_pp": diff.mean() * 100, "dFA_CI95": f"[{lo:+.1f},{hi:+.1f}]",
            "fixed": int(((forced == 1) & (base_correct == 0)).sum()),
            "broken": int(((forced == 0) & (base_correct == 1)).sum()),
            "coverage": ans.mean() * 100, "SA": sel * 100, "SA@80cov": c[:k80].mean() * 100,
            "AURC": aurc, "LLM_calls/q": o.calls.mean(), "retrievals/q": o.retr.mean()}


# ------------------------------------------------------------------------------------------------
# pipelines (each returns Out). `p` = parameters fitted on validation.
# ------------------------------------------------------------------------------------------------
def p_direct(df, p):
    o = Out(len(df)); o.set(np.ones(len(df), bool), df, "direct"); return o


def p_always(df, p, cond):
    o = p_direct(df, p)
    has = df[f"{cond}_pred"].notna().values if f"{cond}_pred" in df else np.zeros(len(df), bool)
    o.set(has, df, cond)
    o.calls += has
    o.retr += 1 if cond.startswith(("rag", "er")) else 0
    if cond.startswith("refl"):
        o.calls[:] = 2
    return o


def p_blind_reflect(df, p, abstain=True):
    o = p_direct(df, p)
    g = np.isin(regime(df, "direct", p["th"]), ["medium", "low"])
    st = f"refl_{p['best_style']}"
    o.set(g, df, st); o.calls += g
    if abstain:
        o.abstain = g & (regime(df, st, p["th"]) == "low")
    return o


def p_final_design(df, p, flag, ctx="filt"):
    """Team's Final Design.xlsx (blind strategy column), given context type, abstain on final low."""
    th, st = p["th"], f"refl_{p['best_style']}"
    o = Out(len(df))
    has_f = df[HAS[ctx]].values.astype(bool)
    reg0 = regime(df, "direct", th)
    reg_rag = regime(df, f"rag_{ctx}", th)
    reg_refl = regime(df, st, th)
    reg_er = regime(df, f"er_{ctx}", th)
    final_reg = reg0.copy()
    for i in range(len(df)):
        if flag[i]:                                   # router says "RAG"
            o.retr[i] = 1
            cur, reg = (f"rag_{ctx}", reg_rag[i]) if has_f[i] else ("direct", reg0[i])
            o.calls[i] += has_f[i]
            if reg != "high":                         # low conf -> retry only
                cur, reg = (f"er_{ctx}", reg_er[i]) if has_f[i] else (st, reg_refl[i])
                o.calls[i] += 1
        else:
            cur, reg = "direct", reg0[i]
            if reg != "high":                         # retry, then reactive RAG on consistent low conf
                cur, reg = st, reg_refl[i]; o.calls[i] += 1
                if reg != "high":
                    o.retr[i] = 1
                    if has_f[i]:
                        cur, reg = f"rag_{ctx}", reg_rag[i]; o.calls[i] += 1
        m = np.zeros(len(df), bool); m[i] = True
        o.set(m, df, cur); final_reg[i] = reg
    o.abstain = final_reg == "low"
    return o


def p_cascade(df, p, abstain=True):
    """Confidence-gated cascade: accept confident direct answers; otherwise try RAG (kept only if it is
    more confident than the direct answer by > delta), then reflection (same acceptance rule)."""
    th, st = p["th"], f"refl_{p['best_style']}"
    o = p_direct(df, p)
    H0 = df.direct_H.values
    unsure = H0 > p["tau"]
    has_f = df[HAS[p["ctx"]]].values.astype(bool)
    rag = f"rag_{p['ctx']}"
    o.retr += unsure
    # RAG step
    try_rag = unsure & has_f
    o.calls += try_rag
    acc = try_rag & (col(df, rag, "margin") > df.direct_margin.values + p["delta"])
    o.set(acc, df, rag)
    # reflection step on still-unsure rows
    still = unsure & (regime_from(o, th) != "high")
    refl = np.where(has_f, "er_" + p["ctx"], st)
    for r in np.unique(refl[still]):
        m = still & (refl == r)
        o.calls += m
        better = m & (col(df, r, "margin") > o.margin + p["delta"])
        o.set(better, df, r)
    if abstain:
        o.abstain = unsure & (regime_from(o, th) == "low")
    return o


def regime_from(o, th):
    return np.array([assign_confidence_regime(h, m, th) for h, m in zip(o.H, o.margin)])


# ------------------------------------------------------------------------------------------------
# learned remedy selector (trained on Train): predicts which pass will be correct
# ------------------------------------------------------------------------------------------------
REMEDIES = ["direct", "rag_raw", "rag_filt", "er_raw", "er_filt",
            "rag_ft_raw", "rag_ft_filt", "er_ft_raw", "er_ft_filt"]


def selector_features(df, post_hoc: bool):
    L = lambda x: np.log(np.asarray(x, float) + 1e-9)
    f = {"logH0": L(df.direct_H), "log1m": L(1 - df.direct_margin + 1e-9), "top1": df.direct_top1.values,
         "s_max": df.s_max.fillna(0).values, "n_filt": df.n_filt.values, "has_raw": df.has_raw.values.astype(float),
         "has_filt": df.has_filt.values.astype(float), "s_max_ft": df.s_max_ft.fillna(0).values,
         "n_filt_ft": df.n_filt_ft.values, "has_raw_ft": df.has_raw_ft.values.astype(float),
         "has_filt_ft": df.has_filt_ft.values.astype(float)}
    for s in ["AQUA-RAT", "ARC-Easy", "MMLU", "MedQA", "ReClor", "SciQ"]:
        f[f"src_{s}"] = (df.source.values == s).astype(float)
    if post_hoc:
        for r in REMEDIES[1:]:
            f[f"{r}_logH"] = L(col(df, r, "H"))
            f[f"{r}_agree"] = (col(df, r, "pred") == df.direct_pred.values).astype(float)
    return pd.DataFrame(f)


def fit_selector(train, post_hoc):
    X = selector_features(train, post_hoc)
    sc = StandardScaler().fit(X)
    models = {}
    for r in REMEDIES:
        y = (col(train, r, "pred") == train.gold.values).astype(int)
        models[r] = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(X), y)
    return sc, models


def p_selector(df, p, key):
    sc, models = p[key]
    post_hoc = key == "sel_post"
    X = sc.transform(selector_features(df, post_hoc))
    P = np.column_stack([models[r].predict_proba(X)[:, 1] for r in REMEDIES])
    # remedies without context are unavailable -> fall back to direct
    avail = np.column_stack([np.ones(len(df), bool)] + [df[f"{r}_pred"].notna().values for r in REMEDIES[1:]])
    P = np.where(avail, P, -1)
    choice = P.argmax(1)
    o = Out(len(df))
    for j, r in enumerate(REMEDIES):
        m = choice == j
        o.set(m, df, r)
    pick_ctx = choice > 0
    o.retr += pick_ctx if not post_hoc else 1
    o.calls += pick_ctx if not post_hoc else avail[:, 1:].sum(1)
    o.margin = P.max(1)                     # selector's own P(correct) is the confidence score
    o.abstain = P.max(1) < p["sel_abstain"]
    return o


def p_oracle_any(df, p):
    o = p_direct(df, p)
    conds = ["direct"] + [f"refl_{s}" for s in PROMPT_STYLE_ORDER] + REMEDIES[1:]
    for c in conds[::-1]:
        ok = (col(df, c, "pred") == df.gold.values)
        o.set(ok, df, c)
    o.calls[:] = len(conds)
    o.retr[:] = 1
    return o


# ------------------------------------------------------------------------------------------------
def fit_params(val, train):
    p = {"th": compute_blind_thresholds(pd.DataFrame({"entropy_over_labels": val.direct_H,
                                                       "margin": val.direct_margin}))}
    base = (val.direct_pred == val.gold).astype(int).values
    # best reflection style for the blind gate, on validation
    best, best_fa = None, -1
    for s in PROMPT_STYLE_ORDER:
        p["best_style"] = s
        fa = (p_blind_reflect(val, p, False).pred == val.gold.values).mean()
        if fa > best_fa:
            best, best_fa = s, fa
    p["best_style"] = best
    # cascade parameters
    grid = []
    for ctx in ["filt", "raw", "ft_filt", "ft_raw"]:
        for q in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for delta in [0.0, 0.1, 0.3]:
                pp = dict(p, ctx=ctx, tau=float(np.quantile(val.direct_H, q)), delta=delta)
                o = p_cascade(val, pp, abstain=False)
                fa = (o.pred == val.gold.values).mean()
                grid.append((fa, -o.calls.mean(), ctx, q, delta))
    fa, _, ctx, q, delta = max(grid)
    p.update(ctx=ctx, tau=float(np.quantile(val.direct_H, q)), tau_q=q, delta=delta)
    if train is not None:
        for key, post in [("sel_pre", False), ("sel_post", True)]:
            p[key] = fit_selector(train, post)
        # abstention threshold for selector: lowest P(correct) giving >= target 95% selective acc on val
        o = p_selector(val, dict(p, sel_abstain=0.0), "sel_post")
        conf, corr = o.margin, (o.pred == val.gold.values)
        p["sel_abstain"] = 0.0
        for t in np.sort(conf):
            if corr[conf >= t].mean() >= 0.95 or (conf >= t).mean() < 0.5:
                p["sel_abstain"] = float(t); break
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("outputs/pipeline_runs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/pipeline_eval"))
    ap.add_argument("--router_dir", type=Path, default=None,
                    help="dir with qwen2.5-{3B,7B}/model/router_model.joblib + intermediate_data/ (original router)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, params_log = [], {}
    for size in SIZES:
        val, test, train = (load(args.runs, size, s) for s in ["validation", "test", "train"])
        if val is None or test is None:
            print(f"[skip] {size}: missing validation/test runs"); continue
        p = fit_params(val, train)
        params_log[size] = {k: v for k, v in p.items() if not k.startswith("sel_") or k == "sel_abstain"}
        base = (test.direct_pred == test.gold).astype(int).values

        # router flag for Final Design: original router (3B/7B) else entropy gate at the router's val flag-rate
        flag, flag_name = None, "entropy-gate"
        if args.router_dir is not None and (args.router_dir / f"qwen2.5-{size}" / "model").exists():
            import joblib
            rd = args.router_dir / f"qwen2.5-{size}"
            r = joblib.load(rd / "model" / "router_model.joblib")
            tm = pd.read_csv(rd / "intermediate_data" / "router_training_matrix_test.csv")
            flag = r.predict(tm[r["imputer"].feature_names_in_.tolist()]).astype(bool)[test.row_id.values]
            flag_name = "original router"
        if flag is None:
            flag = test.direct_H.values > np.quantile(val.direct_H, 0.6)

        pipes = {"Direct": p_direct(test, p),
                 f"Blind-gate reflect ({p['best_style']})": p_blind_reflect(test, p)}
        for src, pre in [("title", ""), ("fulltext", "ft_")]:
            pipes[f"Always RAG [{src}, raw ctx]"] = p_always(test, p, f"rag_{pre}raw")
            pipes[f"Always RAG [{src}, filtered ctx]"] = p_always(test, p, f"rag_{pre}filt")
            o = p_direct(test, p)
            m = flag & test[HAS[pre + "filt"]].values.astype(bool)
            o.set(m, test, f"rag_{pre}filt"); o.calls += m; o.retr += flag
            pipes[f"Router-RAG [{flag_name}; {src}, filtered]"] = o
            pipes[f"Final Design.xlsx [{flag_name}; {src}, filtered]"] = p_final_design(test, p, flag, pre + "filt")
        pipes[f"Confidence-gated cascade (ctx={p['ctx']}, tau=q{p['tau_q']}, delta={p['delta']})"] = p_cascade(test, p)
        if train is not None:
            pipes["Learned selector, pre-hoc (Train-fit)"] = p_selector(test, p, "sel_pre")
            pipes["Learned selector, post-hoc fusion (Train-fit)"] = p_selector(test, p, "sel_post")
        pipes["ORACLE: any pass correct (ceiling)"] = p_oracle_any(test, p)

        for name, o in pipes.items():
            rows.append({"size": size, "pipeline": name, **metrics(o, test, base)})
        cov = {c: int(test[f"{c}_pred"].notna().sum()) for c in REMEDIES[1:] if f"{c}_pred" in test}
        print(f"{size}: context available on test rows -> {cov}")

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 70)
    print(res.round(2).to_string(index=False))
    res.to_csv(args.out / "pipeline_comparison.csv", index=False)
    (args.out / "fitted_params.json").write_text(json.dumps(params_log, indent=2, default=str))
    print(f"\nSaved {args.out / 'pipeline_comparison.csv'}")


if __name__ == "__main__":
    main()
