#!/usr/bin/env python3
"""Summarize completed distance runs without rerunning models or distances."""
import argparse
import json
from pathlib import Path
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "FINDINGS.md")
    args = parser.parse_args()
    lines = ["# Completed metric sensitivity findings", "",
             "These results use the stored Canadian and Scottish participant/summary embeddings. "
             "The PCA basis is fitted on participants only. Each run compares the official summary "
             "with the first 20 archived exact-word-budget random-window references. They are descriptive "
             "geometry checks; windows can be sentence fragments, and they are not coherent LLM summaries.", "",
             "| Dataset / backend | PCA dims | Official W1 | Official W2 | Official MMD² (σ) | "
             "W1–W2 rank ρ | MMD²(σ)–W2 rank ρ | Official closer than random (W2 / MMD²) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    w1_correlations, mmd_correlations, group_rows = [], [], []
    paths = sorted(args.results_root.glob("*/results.json"))
    for path in paths:
        result = json.loads(path.read_text())
        for run, agreement in zip(result["runs"], result["rank_agreement"]):
            official = run["targets"]["official"]
            rank = agreement.get("spearman_rank_vs_w2", {})
            w1, mmd = rank.get("wasserstein_1"), rank.get("mmd2_x1.0")
            if w1 is not None:
                w1_correlations.append(w1)
            if mmd is not None:
                mmd_correlations.append(mmd)
            references = [scores for name, scores in run["targets"].items()
                          if result["target_metadata"][name].get("kind") == "exact_word_budget_random_windows"]
            w2_better = sum(official["wasserstein_2"] < score["wasserstein_2"] for score in references)
            mmd_better = sum(official["mmd2"]["1.0"]["value"] < score["mmd2"]["1.0"]["value"]
                             for score in references)
            lines.append(f"| {result['dataset']} / {result['backend']} | {run['pca_dimensions']} | "
                         f"{official['wasserstein_1']:.4f} | {official['wasserstein_2']:.4f} | "
                         f"{official['mmd2']['1.0']['value']:.4f} | "
                         f"{f'{w1:.3f}' if w1 is not None else '—'} | "
                         f"{f'{mmd:.3f}' if mmd is not None else '—'} | "
                         f"{w2_better}/{len(references)} / {mmd_better}/{len(references)} |")
            groups = run.get("groups", {})
            if len(groups) > 2:
                scores = [group["targets"]["official"] for group in groups.values()]
                group_w2 = [value["wasserstein_2"] for value in scores]
                rho_w1 = float(spearmanr(group_w2, [value["wasserstein_1"] for value in scores]).statistic)
                rho_mmd = float(spearmanr(group_w2, [value["mmd2"]["1.0"] for value in scores]).statistic)
                group_rows.append(f"| {result['dataset']} / {result['backend']} | {run['pca_dimensions']} | "
                                  f"{len(groups)} | {rho_w1:.3f} | {rho_mmd:.3f} |")
    lines.extend(["", f"Completed result files: **{len(paths)}**.", ""])
    if w1_correlations and mmd_correlations:
        lines.extend([
            f"Across these runs W1–W2 target rank agreement ranges from **{min(w1_correlations):.3f} "
            f"to {max(w1_correlations):.3f}**. At the median-source bandwidth, MMD²–W2 agreement "
            f"ranges from **{min(mmd_correlations):.3f} to {max(mmd_correlations):.3f}**.", "",
            "W1/W2 target rankings are very close in these comparisons. MMD preserves much of the "
            "ranking but is not interchangeable with W2. All bandwidth sweeps and group diagnostics "
            "are in each results.json. Do not claim metric-independent equivalence from rank correlation alone.", ""])
    if group_rows:
        lines.extend(["## Frozen-group distances to the official summary", "",
                      "| Dataset / backend | PCA dims | Groups | W1–W2 group rank ρ | MMD²(σ)–W2 group rank ρ |",
                      "|---|---:|---:|---:|---:|", *group_rows, "",
                      "Group rankings can differ considerably between transport and kernel distances "
                      "(especially Education with MPNet at higher PCA dimensions). This is a material "
                      "sensitivity result: distribution distance does not supply a metric-independent "
                      "ordering of undercovered groups. Individual cosine-coverage and lower-tail claims "
                      "are separate quantities and should be evaluated directly, rather than inferred "
                      "from this distribution distance ranking.", ""])
    lines.extend([
        "Distances are compared within each dataset/backend/dimension. The official summaries tend "
        "to be closer than the sampled random-window references, but this says nothing about reaching a "
        "best feasible summary, equitable individual coverage, or correctness of the official interpretation. "
        "Few public summary sentences do not support distribution-test p-values; none are calculated.", "",
        "Results for controlled LLM summaries and Scotland will be added after those targets have been "
        "generated/embedded and evaluated using this same shared-source geometry. No generated-model "
        "findings are implied by these initial Canadian runs.", "",
        "Recreate this report with `python revision/metric_sensitivity/summarize.py`.", ""])
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
