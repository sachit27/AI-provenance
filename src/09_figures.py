#!/usr/bin/env python3
"""Generate figures from one complete analysis output directory."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR


CORRECTED = OUTPUT_DIR
CROSSFIT = CORRECTED
FIGURE_DIR = Path(os.environ.get(
    "PROVENANCE_FIGURE_DIR", OUTPUT_DIR / "figures"
)).resolve()

COLORS = {"education": "#4e79a7", "trust": "#f28e2b"}
MODEL_COLORS = {"openai": "#4e79a7", "mpnet": "#8064a2"}


def style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "legend.frameon": True,
        "legend.framealpha": .85,
        "figure.dpi": 200,
        "savefig.dpi": 350,
        "savefig.bbox": "tight",
        "savefig.pad_inches": .08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
    })


def panel(ax, label: str) -> None:
    ax.text(-0.17, 1.08, label, transform=ax.transAxes, fontsize=24,
            fontweight="bold", va="top", ha="left")


def supplemental_panel(ax, label: str) -> None:
    """Panel labels sized and positioned for denser supplemental composites."""
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="top", ha="left")


def load_topic(topic: str):
    df = pd.read_parquet(CORRECTED / topic / "clustered.parquet")
    transport = json.load(open(CORRECTED / topic / "transport.json"))
    topology = json.load(open(CORRECTED / topic / "topology.json"))
    return df, transport, topology


def make_overview() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for idx, topic in enumerate(["education", "trust"]):
        df, transport, topology = load_topic(topic)
        ax = axes[0, idx]
        labels = sorted(df.cluster_id.unique())
        palette = sns.color_palette("tab10", len(labels))
        summary_2d = np.asarray(topology["summary_2d"])
        medians = []
        for cid, color in zip(labels, palette):
            sub = df[df.cluster_id == cid]
            ax.scatter(sub.umap_x, sub.umap_y, s=6, alpha=.35, color=color,
                       linewidths=0, rasterized=True)
            medians.append((cid, color,
                            float(sub.umap_x.median()), float(sub.umap_y.median())))
        # Collision-aware label placement: nudge each cluster label to the
        # nearest position clear of summary-sentence stars and of previously
        # placed labels (distances normalized by the axis spans). Deterministic.
        xs = np.concatenate([df.umap_x.to_numpy(), summary_2d[:, 0]])
        ys = np.concatenate([df.umap_y.to_numpy(), summary_2d[:, 1]])
        xspan = float(xs.max() - xs.min()) or 1.0
        yspan = float(ys.max() - ys.min()) or 1.0

        def ndist(p, q):
            return float(np.hypot((p[0] - q[0]) / xspan, (p[1] - q[1]) / yspan))

        angles = [k * np.pi / 4 for k in range(8)]
        offsets = [(0.0, 0.0)] + [(np.cos(a) * r, np.sin(a) * r)
                                  for r in (0.06, 0.10, 0.14) for a in angles]
        placed = []
        for cid, color, mx, my in medians:
            cand = (mx, my)
            for ox, oy in offsets:
                trial = (mx + ox * xspan, my + oy * yspan)
                clear_stars = all(ndist(trial, (sx, sy)) >= 0.055
                                  for sx, sy in summary_2d)
                clear_labels = all(ndist(trial, p) >= 0.105 for p in placed)
                if clear_stars and clear_labels:
                    cand = trial
                    break
            placed.append(cand)
            ax.text(cand[0], cand[1], f"C{cid}",
                    fontsize=16, fontweight="bold", ha="center", va="center",
                    zorder=11,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color,
                              lw=1.5, alpha=.95))
        ax.scatter(summary_2d[:, 0], summary_2d[:, 1], marker="*", s=260,
                   color="black", edgecolor="white", linewidth=1.2, zorder=10,
                   label="Summary sentence")
        title = "Education & Skills" if topic == "education" else "Safe AI & Public Trust"
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(loc="best", fontsize=11)
        panel(ax, chr(ord("a") + idx))

    ax = axes[1, 0]
    for topic in ["education", "trust"]:
        df, transport, _ = load_topic(topic)
        topic_label = "Education & Skills" if topic == "education" else "Safe AI & Public Trust"
        ax.hist(df.coverage_score, bins=50, color=COLORS[topic], alpha=.55,
                edgecolor="white", linewidth=.3, label=topic_label)
        ax.axvline(transport["coverage"]["exclusion_threshold"],
                   color=COLORS[topic], linestyle="--", linewidth=2.2,
                   label=f"{'Education' if topic == 'education' else 'Trust'} exclusion $\\tau$ = {transport['coverage']['exclusion_threshold']:.2f}")
        ax.axvline(transport["coverage"]["mean"], color=COLORS[topic],
                   linestyle=":", linewidth=2.2,
                   label=f"{'Education' if topic == 'education' else 'Trust'} mean = {transport['coverage']['mean']:.2f}")
    ax.set_xlabel("Cosine coverage score")
    ax.set_ylabel("Number of participants")
    ax.set_title("Coverage score distributions")
    ax.legend(loc="upper left", fontsize=12)
    panel(ax, "c")

    ax = axes[1, 1]
    for topic in ["education", "trust"]:
        _, transport, _ = load_topic(topic)
        g = transport["gini"]["value"]
        x_lorenz = np.asarray(transport["lorenz_curve"]["x"])
        y_lorenz = np.asarray(transport["lorenz_curve"]["y"])
        label = ("Education & Skills" if topic == "education" else "Safe AI & Public Trust") + f" (Gini = {g:.3f})"
        ax.fill_between(x_lorenz, y_lorenz, x_lorenz, alpha=.18, color=COLORS[topic])
        ax.plot(x_lorenz, y_lorenz, color=COLORS[topic], linewidth=2.8, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=.7,
            linewidth=1.2, label="Perfect equality")
    ax.set_xlabel("Cumulative share of participants")
    ax.set_ylabel("Cumulative share of coverage")
    ax.set_title("Lorenz curves of coverage scores")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=13)
    panel(ax, "d")

    fig.subplots_adjust(hspace=.35, wspace=.25)
    fig.savefig(FIGURE_DIR / "fig2_overview.png")
    plt.close(fig)


def make_clusters() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for col, topic in enumerate(["education", "trust"]):
        _, transport, _ = load_topic(topic)
        stats = pd.DataFrame.from_dict(transport["per_cluster"], orient="index")
        stats.index = stats.index.astype(int)
        stats = stats.sort_values("exclusion_rate", ascending=False)
        y = np.arange(len(stats))
        labels = [f"C{i}" for i in stats.index]

        ax = axes[0, col]
        rates = 100 * stats.exclusion_rate
        bar_colors = ["#d62728" if v >= 80 else "#e15759" if v >= 40
                      else "#f28e2b" if v >= 15 else "#4e79a7" for v in rates]
        bars = ax.barh(y, rates, color=bar_colors, alpha=.9,
                       edgecolor="black", linewidth=.4)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Exclusion rate (%)")
        title = "Education & Skills" if topic == "education" else "Safe AI & Public Trust"
        ax.set_title(title)
        for bar, value in zip(bars, rates):
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                    f"{value:.1f}", va="center", fontsize=10)
        overall = 100 * transport["coverage"]["exclusion_rate"]
        ax.axvline(overall, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Overall = {overall:.1f}%")
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", fontsize=10)
        panel(ax, "a" if topic == "education" else "b")

        ax = axes[1, col]
        errors = 1.96 * stats.std_coverage / np.sqrt(stats.n)
        ax.barh(y, stats.mean_coverage, xerr=errors, color="#4e79a7", alpha=.9,
                edgecolor="black", linewidth=.4, capsize=4)
        threshold = transport["coverage"]["exclusion_threshold"]
        ax.axvline(threshold, color="#e15759", linestyle="--", linewidth=1.8,
                   label=f"Exclusion threshold = {threshold:.2f}")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Mean coverage score")
        ax.set_title(title)
        ax.set_xlim(0, max(.75, float(stats.mean_coverage.max())) * 1.15)
        ax.legend(loc="lower right", fontsize=10)
        panel(ax, "c" if topic == "education" else "d")
    fig.subplots_adjust(hspace=.34, wspace=.24)
    fig.savefig(FIGURE_DIR / "fig3_clusters.png")
    plt.close(fig)


def make_associations() -> None:
    """Corrected replacement for the superseded six-panel AIPW figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    contrast_order = [
        "short_response", "semantically_isolated",
        "hedged_register", "assertive_register",
    ]
    contrast_labels = {
        "short_response": "Short response\n(bottom quintile)",
        "semantically_isolated": "Semantic isolation\n(top two quintiles)",
        "hedged_register": "Hedged register",
        "assertive_register": "Assertive register",
    }
    coef_order = [
        "semantic_isolation", "log_word_count",
        "hedge_rate", "assertiveness_rate",
    ]
    coef_labels = {
        "semantic_isolation": "Semantic isolation",
        "log_word_count": "Log word count",
        "hedge_rate": "Hedge rate",
        "assertiveness_rate": "Assertiveness rate",
    }

    for col, topic in enumerate(["education", "trust"]):
        payload = json.load(open(CORRECTED / topic / "associations.json"))
        topic_label = "Education & Skills" if topic == "education" else "Safe AI & Public Trust"

        contrasts = payload["group_contrasts"]
        vals = np.array([contrasts[k]["difference"] for k in contrast_order])
        low = np.array([contrasts[k]["bootstrap_ci_95"][0] for k in contrast_order])
        high = np.array([contrasts[k]["bootstrap_ci_95"][1] for k in contrast_order])
        y = np.arange(len(contrast_order))
        ax = axes[0, col]
        ax.barh(
            y, vals, xerr=np.vstack([vals-low, high-vals]),
            color=["#e15759" if v < 0 else "#4e79a7" for v in vals],
            alpha=.9, edgecolor="black", linewidth=.4, capsize=4,
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y, [contrast_labels[k] for k in contrast_order])
        ax.set_xlabel("Observed mean coverage difference")
        ax.set_title(f"Descriptive contrasts: {topic_label}")
        panel(ax, "a" if col == 0 else "b")

        ols = payload["continuous_ols_hc3"]
        coefs = ols["coefficients"]
        vals = np.array([coefs[k]["coefficient"] for k in coef_order])
        low = np.array([coefs[k]["ci_95"][0] for k in coef_order])
        high = np.array([coefs[k]["ci_95"][1] for k in coef_order])
        ax = axes[1, col]
        ax.barh(
            y, vals, xerr=np.vstack([vals-low, high-vals]),
            color=["#e15759" if v < 0 else "#4e79a7" for v in vals],
            alpha=.9, edgecolor="black", linewidth=.4, capsize=4,
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y, [coef_labels[k] for k in coef_order])
        ax.set_xlabel("Standardized OLS coefficient ($\\beta$)")
        ax.set_title(f"Exploratory OLS: {topic_label} ($R^2={ols['r_squared']:.3f}$)")
        panel(ax, "c" if col == 0 else "d")

    fig.subplots_adjust(hspace=.42, wspace=.30)
    fig.savefig(FIGURE_DIR / "fig4_associations.png")
    plt.close(fig)


def make_benchmarks() -> None:
    data = pd.read_csv(CROSSFIT / "crossfit_confirmatory_summary.csv")
    data["label"] = data.topic.str.title() + "\n" + data.model.map({"openai": "OpenAI", "mpnet": "MPNet"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(len(data))
    width = .36

    ax = axes[0, 0]
    ax.bar(x - width/2, data.exact_random_null_mean, width, color="#BDBDBD",
           edgecolor="black", linewidth=.4, label="Exact-length random")
    ax.bar(x + width/2, data.exact_random_official_mean, width,
           color=[MODEL_COLORS[m] for m in data.model], edgecolor="black",
           linewidth=.4, label="Official")
    for i, p in enumerate(data.exact_random_mean_p):
        ax.text(i, max(data.exact_random_null_mean.iloc[i], data.exact_random_official_mean.iloc[i]) + .012,
                f"p={p:.3f}", ha="center", fontsize=11)
    ax.set_xticks(x, data.label)
    ax.set_ylim(.35, .72)
    ax.set_ylabel("Mean coverage")
    ax.set_title("Exact-length chance benchmark")
    ax.legend(ncol=2)
    panel(ax, "a")

    ax = axes[0, 1]
    ax.bar(x - width/2, data.official_mean_coverage, width, color="#BDBDBD",
           edgecolor="black", linewidth=.4, label="Official")
    ax.bar(x + width/2, data.mean_optimized_coverage, width,
           color=[MODEL_COLORS[m] for m in data.model], edgecolor="black",
           linewidth=.4, label="Mean-optimized")
    ax.set_xticks(x, data.label)
    ax.set_ylim(.42, .74)
    ax.set_ylabel("Held-out mean coverage")
    ax.set_title("Budget-matched feasible frontier")
    ax.legend(ncol=2)
    panel(ax, "b")

    ax = axes[1, 0]
    ax.bar(x - width/2, data.official_bottom_decile_mean, width, color="#BDBDBD",
           edgecolor="black", linewidth=.4, label="Official")
    ax.bar(x + width/2, data.tail_optimized_bottom_decile_mean, width,
           color=[MODEL_COLORS[m] for m in data.model], edgecolor="black",
           linewidth=.4, label="Tail-optimized")
    ax.set_xticks(x, data.label)
    ax.set_ylim(.22, .46)
    ax.set_ylabel("Held-out bottom-decile mean")
    ax.set_title("Lower-tail coverage")
    ax.legend(ncol=2)
    panel(ax, "c")

    ax = axes[1, 1]
    ax.bar(x - width/2, 100 * data.official_exclusion_rate, width, color="#BDBDBD",
           edgecolor="black", linewidth=.4, label="Official")
    ax.bar(x + width/2, 100 * data.tail_optimized_exclusion_rate, width,
           color=[MODEL_COLORS[m] for m in data.model], edgecolor="black",
           linewidth=.4, label="Tail-optimized")
    ax.set_xticks(x, data.label)
    ax.set_ylim(0, 22)
    ax.set_ylabel("Low coverage at official-reference threshold (%)")
    ax.set_title("Low-coverage rate")
    ax.legend(ncol=2)
    panel(ax, "d")

    fig.subplots_adjust(hspace=.38, wspace=.25)
    fig.savefig(FIGURE_DIR / "fig5_benchmarks.png")
    plt.close(fig)


def make_robustness() -> None:
    comparison = json.load(open(CORRECTED / "cross_topic" / "comparison.json"))
    consistency = comparison["participant_consistency"]
    contingency = consistency["exclusion_contingency"]
    e, _, _ = load_topic("education")
    t, _, _ = load_topic("trust")
    overlap = e[["Internal ID", "coverage_score", "is_excluded"]].merge(
        t[["Internal ID", "coverage_score", "is_excluded"]], on="Internal ID",
        suffixes=("_education", "_trust")
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    sns.regplot(data=overlap, x="coverage_score_education", y="coverage_score_trust",
                scatter_kws={"s": 10, "alpha": .20},
                line_kws={"color": "black", "linewidth": 1}, ax=ax)
    ax.set_xlabel("Education coverage")
    ax.set_ylabel("Trust coverage")
    ax.set_title(
        "Within-participant consistency "
        f"($\\rho_s={consistency['coverage_spearman_rho']:.3f}$)"
    )
    panel(ax, "a")

    ax = axes[0, 1]
    table = np.array([
        [contingency["both"], contingency["only_education"]],
        [contingency["only_trust"], contingency["neither"]],
    ])
    odds_ratio = (
        contingency["both"] * contingency["neither"] /
        (contingency["only_education"] * contingency["only_trust"])
    )
    sns.heatmap(table, annot=True, fmt=",", cmap="YlOrRd", cbar=False, ax=ax,
                annot_kws={"fontsize": 14, "fontweight": "bold"},
                xticklabels=["Trust\nlow coverage", "Trust\nnot low coverage"],
                yticklabels=["Education\nlow coverage", "Education\nnot low coverage"])
    ax.set_title(f"Cross-topic status (OR={odds_ratio:.2f})")
    panel(ax, "b")

    ax = axes[1, 0]
    rows = []
    for topic, models in comparison["multi_model_robustness"].items():
        for model, vals in models.items():
            rows.append((topic, model, vals["coverage_rank_correlation"], vals["exclusion_agreement"]))
    rr = pd.DataFrame(rows, columns=["topic", "model", "rho", "agreement"])
    rr["label"] = rr.topic.str.title() + "\n" + rr.model.map({"all-mpnet-base-v2": "MPNet", "nomic-embed-text": "Nomic"})
    ax.bar(np.arange(len(rr)), rr.rho, color=[COLORS[t] for t in rr.topic],
           edgecolor="black", linewidth=.4)
    ax.set_xticks(np.arange(len(rr)), rr.label)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title("Embedding-model rank robustness")
    panel(ax, "c")

    ax = axes[1, 1]
    for topic in ["education", "trust"]:
        sweep = comparison["parameter_sensitivity"][topic]["exclusion_sweep"]
        ax.plot([r["sd_multiplier"] for r in sweep],
                [100 * r["exclusion_rate"] for r in sweep], marker="o",
                color=COLORS[topic], linewidth=2,
                label="Education" if topic == "education" else "Trust")
    ax.set_xlabel("Threshold multiplier below mean (SD)")
    ax.set_ylabel("Excluded (%)")
    ax.set_title("Threshold sensitivity")
    ax.legend()
    panel(ax, "d")

    fig.subplots_adjust(hspace=.38, wspace=.28)
    fig.savefig(FIGURE_DIR / "fig6_robustness.png")
    plt.close(fig)


def make_sensitivity() -> None:
    comparison = json.load(open(CORRECTED / "cross_topic" / "comparison.json"))
    with plt.rc_context({
        "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
        for topic in ["education", "trust"]:
            sens = comparison["parameter_sensitivity"][topic]
            color = COLORS[topic]
            label = "Education" if topic == "education" else "Trust"
            axes[0, 0].plot([x["pca_dims"] for x in sens["pca_sweep"]],
                            [x["wasserstein_2"] for x in sens["pca_sweep"]],
                            marker="o", linewidth=2, color=color, label=label)
            axes[0, 1].plot([x["pca_dims"] for x in sens["pca_sweep"]],
                            [x["gini"] for x in sens["pca_sweep"]],
                            marker="o", linewidth=2, color=color, label=label)
            axes[1, 0].plot([x["sd_multiplier"] for x in sens["exclusion_sweep"]],
                            [100*x["exclusion_rate"] for x in sens["exclusion_sweep"]],
                            marker="o", linewidth=2, color=color, label=label)
            axes[1, 1].plot([x["k"] for x in sens["k_sweep"]],
                            [x["silhouette"] for x in sens["k_sweep"]],
                            marker="o", linewidth=2, color=color, label=label)
        titles = ["Wasserstein distance", "Coverage Gini", "Low-coverage rate", "Cluster silhouette"]
        ylabels = ["$W_2$", "Gini", "Low coverage (%)", "Silhouette"]
        xlabels = ["PCA dimensions", "PCA dimensions", "Threshold multiplier (SD)", "$k$"]
        for i, ax in enumerate(axes.flat):
            ax.set_title(titles[i]); ax.set_ylabel(ylabels[i]); ax.set_xlabel(xlabels[i])
            ax.legend(frameon=False); supplemental_panel(ax, chr(ord("a") + i))
        axes[0, 1].set_ylim(.10, .15)
        fig.subplots_adjust(hspace=.42, wspace=.30, left=.09, right=.98, top=.95, bottom=.09)
        fig.savefig(FIGURE_DIR / "sfig3_sensitivity.png")
        plt.close(fig)


def make_cluster_selection_supplement() -> None:
    """Generate Figure S1 directly from the current topology outputs."""
    with plt.rc_context({
        "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
        for column, topic in enumerate(("education", "trust")):
            topology = json.load(open(CORRECTED / topic / "topology.json"))
            kmeans = topology["kmeans"]
            scores = kmeans["silhouette_scores"]
            ks = sorted(int(value) for value in scores)
            ax = axes[0, column]
            ax.plot(
                ks, [scores[str(k)] for k in ks], "o-",
                color=COLORS[topic], linewidth=2, markersize=5,
                label="Silhouette",
            )
            selected = int(kmeans["best_k"])
            ax.axvline(
                selected, color="black", linestyle="--", linewidth=1.2,
                label=f"Selected k = {selected}",
            )
            stability = kmeans["bootstrap_stability"]
            ax.text(
                .97, .05,
                f"Bootstrap ARI = {stability['mean_ari']:.3f}\n"
                f"({100 * stability['pct_above_0.8']:.0f}% iterations ≥ 0.80)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "gray", "lw": .5},
            )
            ax.set_xlabel("Number of clusters $k$")
            ax.set_ylabel("Silhouette coefficient")
            ax.set_title("Education & Skills" if topic == "education" else "Safe AI & Public Trust")
            ax.legend(loc="upper right", frameon=True)
            supplemental_panel(ax, "a" if column == 0 else "b")

            ax = axes[1, column]
            coherence = topology["topic_coherence"]
            cluster_ids = sorted(coherence["per_cluster"], key=int)
            values = [coherence["per_cluster"][cluster] for cluster in cluster_ids]
            palette = sns.color_palette("muted", n_colors=max(9, len(values)))
            ax.bar(
                np.arange(len(values)), values,
                color=[palette[int(cluster) % len(palette)] for cluster in cluster_ids],
                edgecolor="black", linewidth=.3,
            )
            mean_npmi = float(coherence["mean_npmi"])
            ax.axhline(
                mean_npmi, color="black", linestyle="--", linewidth=1.2,
                label=f"Mean NPMI = {mean_npmi:.3f}",
            )
            ax.set_xticks(np.arange(len(values)), [f"C{cluster}" for cluster in cluster_ids])
            ax.set_ylabel("NPMI")
            ax.set_title("Education & Skills" if topic == "education" else "Safe AI & Public Trust")
            ax.legend(loc="lower right", frameon=True)
            supplemental_panel(ax, "c" if column == 0 else "d")
        fig.subplots_adjust(hspace=.42, wspace=.30, left=.09, right=.98, top=.95, bottom=.09)
        fig.savefig(FIGURE_DIR / "sfig1_cluster_selection.png")
        plt.close(fig)


def make_crossfit_supplement() -> None:
    summary = pd.read_csv(CROSSFIT / "crossfit_confirmatory_summary.csv")
    fold_rows, repeat_rows = [], []
    for topic in ["education", "trust"]:
        for model in ["openai", "mpnet"]:
            payload = json.load(open(CROSSFIT / topic / f"crossfit_{model}.json"))
            label = f"{topic.title()}\n{'OpenAI' if model == 'openai' else 'MPNet'}"
            for fold in payload["fold_results"]:
                metrics = fold["metrics"]
                fold_rows.append({
                    "label": label,
                    "mean_difference": metrics["mean_optimized_extractive"]["mean_coverage"] - metrics["official"]["mean_coverage"],
                    "tail_difference": metrics["balanced_tail_optimized_extractive"]["bottom_decile_mean"] - metrics["official"]["bottom_decile_mean"],
                })
            rep = payload["repeat_level_sensitivity"]["paired_vs_official"]
            for value in rep["mean_optimized_extractive"]["mean_coverage"]["repeat_differences"]:
                repeat_rows.append({"label": label, "metric": "Mean", "difference": value})
            for value in rep["balanced_tail_optimized_extractive"]["bottom_decile_mean"]["repeat_differences"]:
                repeat_rows.append({"label": label, "metric": "Bottom decile", "difference": value})
    folds = pd.DataFrame(fold_rows)
    repeats = pd.DataFrame(repeat_rows)
    label_map = {
        "Education\nOpenAI": "Edu.\nOpenAI", "Education\nMPNet": "Edu.\nMPNet",
        "Trust\nOpenAI": "Trust\nOpenAI", "Trust\nMPNet": "Trust\nMPNet",
    }
    folds["plot_label"] = folds.label.map(label_map)
    repeats["plot_label"] = repeats.label.map(label_map)
    order = list(label_map.values())

    with plt.rc_context({
        "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
        sns.boxplot(data=folds, x="plot_label", y="mean_difference", order=order,
                    color="#4C78A8", width=.6, ax=axes[0, 0])
        axes[0, 0].axhline(0, color="black", linewidth=1)
        axes[0, 0].set_title("Fold-level mean-coverage gains")
        axes[0, 0].set_xlabel(""); axes[0, 0].set_ylabel("Optimized minus official")
        supplemental_panel(axes[0, 0], "a")

        sns.boxplot(data=folds, x="plot_label", y="tail_difference", order=order,
                    color="#F58518", width=.6, ax=axes[0, 1])
        axes[0, 1].axhline(0, color="black", linewidth=1)
        axes[0, 1].set_title("Fold-level bottom-decile gains")
        axes[0, 1].set_xlabel(""); axes[0, 1].set_ylabel("Optimized minus official")
        supplemental_panel(axes[0, 1], "b")

        ax = axes[1, 0]
        x = np.arange(len(summary)); w = .36
        ax.bar(x-w/2, 100*summary.worst_cluster_official_exclusion, w,
               color="#BDBDBD", edgecolor="black", linewidth=.4, label="Official")
        ax.bar(x+w/2, 100*summary.worst_cluster_tail_optimized_exclusion, w,
               color=[MODEL_COLORS[m] for m in summary.model], edgecolor="black",
               linewidth=.4, label="Tail-optimized")
        ax.set_xticks(x, order); ax.set_ylim(0, 100)
        ax.set_ylabel("Worst-cluster low coverage (%)")
        ax.set_title("Worst-cluster diagnostic")
        ax.legend(frameon=False); supplemental_panel(ax, "c")

        ax = axes[1, 1]
        sns.stripplot(data=repeats, x="plot_label", y="difference", hue="metric",
                      order=order, dodge=True, jitter=.08, size=5,
                      palette={"Mean": "#4C78A8", "Bottom decile": "#F58518"}, ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel(""); ax.set_ylabel("Optimized minus official")
        ax.set_title("Five complete cross-fit repetitions")
        ax.legend(frameon=False, title=""); supplemental_panel(ax, "d")

        fig.subplots_adjust(hspace=.48, wspace=.30, left=.09, right=.98, top=.95, bottom=.10)
        fig.savefig(FIGURE_DIR / "sfig2_crossfit.png")
        plt.close(fig)


def make_intersections_supplement() -> None:
    """Descriptive short-by-isolation cells replacing old interaction claims."""
    with plt.rc_context({
        "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
        for idx, topic in enumerate(["education", "trust"]):
            payload = json.load(open(CORRECTED / topic / "associations.json"))
            cells = payload["descriptive_intersections"]
            aggregates = {}
            for short in [0, 1]:
                for isolated in [0, 1]:
                    selected = [r for r in cells if r["short"] == short and r["isolated"] == isolated]
                    n = sum(r["n"] for r in selected)
                    mean = sum(r["n"] * r["mean_coverage"] for r in selected) / n
                    ss = sum(
                        (r["n"] - 1) * (r["sd_coverage"] or 0) ** 2
                        + r["n"] * (r["mean_coverage"] - mean) ** 2
                        for r in selected
                    )
                    sd = np.sqrt(ss / (n - 1))
                    aggregates[(short, isolated)] = (n, mean, 1.96 * sd / np.sqrt(n))

            ax = axes[idx]
            x = np.arange(2); width = .34
            for isolated, offset, color, label in [
                (0, -width/2, "#4e79a7", "Not isolated"),
                (1, width/2, "#e15759", "Semantically isolated"),
            ]:
                means = [aggregates[(short, isolated)][1] for short in [0, 1]]
                errs = [aggregates[(short, isolated)][2] for short in [0, 1]]
                bars = ax.bar(x + offset, means, width, yerr=errs, capsize=4,
                              color=color, edgecolor="black", linewidth=.4, label=label)
                for short, bar in enumerate(bars):
                    n = aggregates[(short, isolated)][0]
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+errs[short]+.008,
                            f"n={n}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x, ["Not short", "Short response"])
            ax.set_ylim(.30, .66)
            ax.set_ylabel("Mean coverage")
            ax.set_title("Education & Skills" if topic == "education" else "Safe AI & Public Trust")
            ax.legend(frameon=False, loc="upper right")
            supplemental_panel(ax, "a" if idx == 0 else "b")
        fig.subplots_adjust(wspace=.28, left=.08, right=.98, top=.90, bottom=.15)
        fig.savefig(FIGURE_DIR / "sfig4_intersections.png")
        plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    style()
    make_overview()
    make_clusters()
    make_associations()
    make_benchmarks()
    make_robustness()
    make_cluster_selection_supplement()
    make_crossfit_supplement()
    make_sensitivity()
    make_intersections_supplement()
    print("Generated figures in", FIGURE_DIR)


if __name__ == "__main__":
    main()
