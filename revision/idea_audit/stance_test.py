"""Are opposition/concern ideas missed more often than other ideas, holding prevalence fixed?
Official summary and every generated summary; ideas with >=10 respondents (primary) and >=5 (sensitivity).
Outcome: final label == MISSED. Model: logit(MISSED) ~ oppose_concern + log(n_resp) + topic, cluster-robust SE by idea.

Details
- Generated summaries = summary names other than 'official' and not starting with 'official' (so 'official_wider'
  is excluded from both groups).
- Final label: an official-summary idea labelled MISSED that the wider-findings judgment ('official_wider') labels
  COVERED is RELOCATED, which is not MISSED.
- A logit fit that does not converge (e.g. complete separation) is reported as 'not estimable', never as numbers.
- Fisher exact tests treat idea x summary pairs as independent, so they are reported for the official summary only
  (one judgment per idea); for generated summaries the clustered model is the inference.
Writes STANCE_TEST.md and stance_test_results.csv next to this script."""
from pathlib import Path
import warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
W = HERE / "work"
NOT_EST = "not estimable (no convergence/separation)"

rows = []
for t in ["education", "trust", "q7", "q21"]:
    ideas = pd.read_parquet(W / t / "ideas_t0.30.parquet")[["idea_id", "stance", "n_resp"]]
    j = pd.read_parquet(W / t / "judgments_t0.30.parquet")[["idea_id", "summary", "label"]].copy()
    wider = j[j.summary == "official_wider"].set_index("idea_id").label
    reloc = (j.summary == "official") & (j.label == "MISSED") & (j.idea_id.map(wider) == "COVERED")
    j.loc[reloc, "label"] = "RELOCATED"
    j = j[j.summary != "official_wider"]
    d = j.merge(ideas, on="idea_id"); d["topic"] = t; rows.append(d)
D = pd.concat(rows, ignore_index=True)
D["missed"] = (D.label == "MISSED").astype(int)
D["opp"] = D.stance.isin(["oppose", "concern"]).astype(int)
D["log_n"] = np.log(D.n_resp)
D["uid"] = D.topic + ":" + D.idea_id.astype(str)
D["source"] = np.where(D.summary == "official", "official", "generated")
# Q21 generated summaries are excluded: 7 of 9 missed the length budget, so Q21 is not part of the controlled comparison.
D = D[~((D.source == "generated") & (D.topic == "q21"))]
assert not D.summary.str.startswith("official_").any()

out = ["# Opposition/concern ideas vs other ideas: probability of MISSED\n",
       "Logit(MISSED) ~ oppose/concern + log(n respondents) + topic; SEs clustered by idea. Fisher exact tests are shown for the official summary only "
       "(for generated summaries idea x summary pairs are not independent, so only the clustered model is the inference).\n"]
recs = []
for min_resp in (10, 5):
    for src in ("official", "generated"):
        for scope, topics in (("Canada", ["education", "trust"]), ("all four topics (generated: Q21 excluded)", ["education", "trust", "q7", "q21"])):
            s = D[(D.n_resp >= min_resp) & (D.source == src) & D.topic.isin(topics)]
            n_opp = int((s.opp == 1).sum()); n_oth = int((s.opp == 0).sum())
            k_opp = int(s[s.opp == 1].missed.sum()); k_oth = int(s[s.opp == 0].missed.sum())
            f = "missed ~ opp + log_n" + (" + C(topic)" if s.topic.nunique() > 1 else "")
            status, orr, lo, hi, p = "ok", np.nan, np.nan, np.nan, np.nan
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = smf.logit(f, s).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(s.uid)[0]})
                b, se, p = m.params["opp"], m.bse["opp"], m.pvalues["opp"]
                if not m.mle_retvals.get("converged", False) or not np.isfinite([b, se, p]).all():
                    status = "not_estimable"
                else:
                    orr, lo, hi = np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
            except Exception:
                status = "not_estimable"
            if status != "ok":
                orr = lo = hi = p = np.nan
            fe_or = fe_p = np.nan
            if src == "official":
                tab = np.array([[k_opp, n_opp - k_opp], [k_oth, n_oth - k_oth]])
                fe_or, fe_p = fisher_exact(tab)
            recs.append(dict(min_resp=min_resp, summaries=src, scope=scope, pairs=len(s), ideas=s.uid.nunique(),
                             pct_missed_opp=100 * k_opp / n_opp, k_opp=k_opp, n_opp=n_opp,
                             pct_missed_other=100 * k_oth / n_oth, k_oth=k_oth, n_oth=n_oth,
                             fisher_or=fe_or, fisher_p=fe_p, adj_or=orr, adj_lo=lo, adj_hi=hi, adj_p=p, status=status))
            res = (f"{orr:.2f} [{lo:.2f}, {hi:.2f}]" if status == "ok" else NOT_EST)
            pp = f"{p:.2g}" if status == "ok" else "n/a"
            fis = f"{fe_or:.2f} ({fe_p:.2g})" if src == "official" else "not reported (pairs not independent)"
            out.append(f"\n## >= {min_resp} respondents | {src} summaries | {scope}\n\n"
                       f"| ideas×summaries | %MISSED oppose/concern | %MISSED other | Fisher OR (p) | adjusted OR [95% CI] | adj. p |\n|---|---|---|---|---|---|\n"
                       f"| {len(s)} ({s.uid.nunique()} ideas) | {100*k_opp/n_opp:.1f} ({k_opp}/{n_opp}) | {100*k_oth/n_oth:.1f} ({k_oth}/{n_oth}) | {fis} | {res} | {pp} |\n")
pd.DataFrame(recs).to_csv(HERE / "stance_test_results.csv", index=False)
(HERE / "STANCE_TEST.md").write_text("".join(out))
print("".join(out))
