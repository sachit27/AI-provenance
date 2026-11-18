#!/usr/bin/env python3
"""05_associations.py - descriptive and associational coverage analysis.

This script does *not* estimate causal effects. Earlier versions used AIPW for
binary indicators that were deterministic functions of included propensity
covariates, violating conditional positivity. That estimand has been removed.

The replacement analysis reports:
  1. Observed mean coverage differences for four pre-specified text groups,
     with stratified bootstrap confidence intervals and Welch p-values.
  2. Exploratory multivariable OLS associations for continuous text features,
     with HC3 heteroskedasticity-robust standard errors.
  3. A cluster-fixed-effect sensitivity model and descriptive intersections.

These quantities characterize the analysed records only. They do not identify
effects of changing how a participant writes and are not generalized to the
population of Canada.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import (
    BOOTSTRAP_B,
    ISOLATION_QUINTILE_BINS,
    NumpyEncoder,
    OUTPUT_DIR,
    TOPICS,
    WORD_COUNT_QUINTILE_BINS,
)

SEED = 42


def bootstrap_mean_difference(y, group, n_boot=BOOTSTRAP_B, seed=SEED):
    """Treated-minus-comparison mean difference with stratified bootstrap CI."""
    y = np.asarray(y, dtype=float)
    group = np.asarray(group, dtype=bool)
    y1, y0 = y[group], y[~group]
    if len(y1) < 2 or len(y0) < 2:
        raise ValueError("Both groups require at least two observations")
    rng = np.random.RandomState(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draws[b] = (
            rng.choice(y1, size=len(y1), replace=True).mean()
            - rng.choice(y0, size=len(y0), replace=True).mean()
        )
    diff = float(y1.mean() - y0.mean())
    ci = np.percentile(draws, [2.5, 97.5])
    welch = stats.ttest_ind(y1, y0, equal_var=False)
    pooled_sd = np.sqrt(((len(y1) - 1) * y1.var(ddof=1) +
                         (len(y0) - 1) * y0.var(ddof=1)) /
                        (len(y1) + len(y0) - 2))
    return {
        "estimand": "observed mean coverage difference (group minus comparison)",
        "difference": diff,
        "bootstrap_ci_95": [float(ci[0]), float(ci[1])],
        "bootstrap_b": int(n_boot),
        "welch_p_value_two_sided": float(welch.pvalue),
        "standardized_mean_difference": float(diff / pooled_sd) if pooled_sd > 0 else None,
        "group_mean": float(y1.mean()),
        "comparison_mean": float(y0.mean()),
        "group_sd": float(y1.std(ddof=1)),
        "comparison_sd": float(y0.std(ddof=1)),
        "n_group": int(len(y1)),
        "n_comparison": int(len(y0)),
    }


def fit_hc3(y, features):
    """Fit OLS with standardized predictors and HC3 standard errors."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)
    model = sm.OLS(y, sm.add_constant(x_scaled)).fit(cov_type="HC3")
    coefficients = {}
    for i, name in enumerate(["const"] + list(features.columns)):
        coefficients[name] = {
            "coefficient": float(model.params[i]),
            "hc3_se": float(model.bse[i]),
            "ci_95": [float(model.conf_int()[i, 0]), float(model.conf_int()[i, 1])],
            "p_value_two_sided": float(model.pvalues[i]),
        }
    vifs = {
        name: float(variance_inflation_factor(x_scaled, i))
        for i, name in enumerate(features.columns)
    }
    return {
        "model": "OLS with standardized predictors and HC3 robust standard errors",
        "interpretation": "coverage-score difference associated with a one-SD predictor difference, conditional on the other listed predictors",
        "coefficients": coefficients,
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "vif": vifs,
        "n": int(model.nobs),
    }


def process_topic(topic_name):
    topic_dir = OUTPUT_DIR / topic_name
    print(f"\n{'=' * 60}\nAssociational analysis: {topic_name}\n{'=' * 60}")
    df = pd.read_parquet(topic_dir / "clustered.parquet").copy()
    if "coverage_score" not in df:
        raise KeyError("coverage_score must be added by 04_transport.py before step 05")
    y = df["coverage_score"].to_numpy(dtype=float)

    # Pre-specified descriptive groups. These labels are deterministic summaries
    # of observed text, not manipulable treatments.
    df["word_count_quintile"] = pd.qcut(
        df["word_count"], WORD_COUNT_QUINTILE_BINS, labels=False, duplicates="drop"
    )
    df["isolation_quintile"] = pd.qcut(
        df["semantic_isolation"], ISOLATION_QUINTILE_BINS,
        labels=False, duplicates="drop"
    )
    groups = {
        "short_response": {
            "label": "Short response (bottom word-count quintile)",
            "indicator": (df["word_count_quintile"] == 0).to_numpy(),
        },
        "semantically_isolated": {
            "label": "Semantically isolated (top two isolation quintiles)",
            "indicator": (df["isolation_quintile"] >= 3).to_numpy(),
        },
        "assertive_register": {
            "label": "Assertive lexical register",
            "indicator": (df["register"] == "assertive").to_numpy(),
        },
        "hedged_register": {
            "label": "Hedged lexical register",
            "indicator": (df["register"] == "hedged").to_numpy(),
        },
    }
    contrasts = {}
    for offset, (name, spec) in enumerate(groups.items()):
        result = bootstrap_mean_difference(y, spec["indicator"], seed=SEED + offset)
        result["label"] = spec["label"]
        contrasts[name] = result
        print(
            f"  {spec['label']}: difference={result['difference']:.4f}, "
            f"95% bootstrap CI [{result['bootstrap_ci_95'][0]:.4f}, "
            f"{result['bootstrap_ci_95'][1]:.4f}]"
        )

    # Continuous predictors avoid artificial thresholding in the complementary
    # regression. Word count is log-transformed because of its long right tail.
    features = pd.DataFrame({
        "log_word_count": np.log1p(df["word_count"].astype(float)),
        "semantic_isolation": df["semantic_isolation"].astype(float),
        "hedge_rate": df["hedge_rate"].astype(float),
        "assertiveness_rate": df["assertiveness_rate"].astype(float),
    })
    ols = fit_hc3(y, features)

    # Sensitivity model conditions on data-derived cluster indicators. It is
    # reported separately because clusters and coverage share an embedding space.
    cluster_dummies = pd.get_dummies(
        df["cluster_id"].astype(str), prefix="cluster", drop_first=True, dtype=float
    )
    sensitivity_features = pd.concat([features, cluster_dummies], axis=1)
    ols_cluster_fe = fit_hc3(y, sensitivity_features)

    # Descriptive intersections only: no coefficient is given a causal meaning.
    df["short"] = groups["short_response"]["indicator"].astype(int)
    df["isolated"] = groups["semantically_isolated"]["indicator"].astype(int)
    df["hedged"] = groups["hedged_register"]["indicator"].astype(int)
    intersection = []
    for key, sub in df.groupby(["short", "isolated", "hedged"], observed=True):
        vals = sub["coverage_score"].to_numpy(dtype=float)
        intersection.append({
            "short": int(key[0]),
            "isolated": int(key[1]),
            "hedged": int(key[2]),
            "n": int(len(sub)),
            "mean_coverage": float(vals.mean()),
            "sd_coverage": float(vals.std(ddof=1)) if len(vals) > 1 else None,
        })

    per_cluster = {}
    for cid, sub in df.groupby("cluster_id"):
        per_cluster[str(int(cid))] = {
            "n": int(len(sub)),
            "mean_coverage": float(sub["coverage_score"].mean()),
            "sd_coverage": float(sub["coverage_score"].std(ddof=1)),
            "exclusion_rate": float(sub["is_excluded"].mean()),
        }

    output = {
        "analysis_type": "descriptive_and_associational",
        "causal_effects_estimated": False,
        "scope": (
            "Observed English-language consultation records retained by the "
            "documented preprocessing pipeline; no population-general causal claim."
        ),
        "reason_aipw_removed": (
            "The former binary indicators were deterministic functions of variables "
            "included in their propensity models, so conditional positivity did not hold."
        ),
        "outcome": "maximum cosine similarity to an official-summary sentence",
        "group_contrasts": contrasts,
        "continuous_ols_hc3": ols,
        "cluster_fixed_effect_sensitivity": ols_cluster_fe,
        "descriptive_intersections": intersection,
        "per_cluster": per_cluster,
        "random_seed": SEED,
    }
    with open(topic_dir / "associations.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"  Saved {topic_dir / 'associations.json'}")


def main():
    for topic_name in TOPICS:
        process_topic(topic_name)
    print("\nAssociation analysis complete.")


if __name__ == "__main__":
    main()
