#!/usr/bin/env python3
"""Run W1/W2/Gaussian-MMD sensitivity without contacting an external service."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
from scipy.stats import spearmanr

from metrics import evaluate_targets

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_fingerprint(texts):
    return hashlib.sha256(json.dumps(texts, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def load_target(path):
    with np.load(path, allow_pickle=False) as saved:
        for key in ("summary", "embeddings"):
            if key in saved:
                return np.asarray(saved[key], dtype=float)
    raise ValueError(f"{path}: expected 'summary' or 'embeddings' array")


def load_canada(topic, backend, n_random):
    sys.path.insert(0, str(ROOT / "src"))
    import analysis_io
    from config import OUTPUT_DIR
    if backend == "nomic_embed_text":
        source, official = analysis_io.nomic_core_embeddings(topic)
        blocks = None
    else:
        blocks = analysis_io.embedding_blocks(topic, backend)
        source, official = blocks["records"], blocks["summary"]
    path = OUTPUT_DIR / topic / f"embeddings_{backend}.npz"
    targets = {"official": official}
    descriptions = {"official": {"kind": "official"}}
    if n_random and blocks is not None:
        offset = 0
        for index, draw in enumerate(blocks["null_draws"][:n_random]):
            name = f"exact_budget_random_{index:04d}"
            targets[name] = blocks["null_embeddings"][offset:offset + len(draw)]
            descriptions[name] = {"kind": "exact_word_budget_random_windows",
                                  "words": sum(item["word_count"] for item in draw),
                                  "warning": "Windows may be sentence fragments; geometric reference only"}
            offset += len(draw)
    elif n_random:
        raise ValueError("Nomic archive contains no random-window embeddings; use --random-references 0")
    dependencies = [path, OUTPUT_DIR / topic / "clean.parquet",
                    ROOT / "src" / "config.py", ROOT / "src" / "analysis_io.py"]
    if blocks is not None:
        dependencies.extend([OUTPUT_DIR / topic / "complete_sentence_candidates.parquet",
                             OUTPUT_DIR / topic / "exact_random_windows.json"])
    return source, targets, descriptions, dependencies, f"canada_{topic}"


def load_original(root, pipeline_dir, topic, backend, n_random):
    """Frozen outputs of a re-run of the original pipeline (e.g. Scotland).

    ``root`` is that run's analysis directory and ``pipeline_dir`` its copy of
    the pipeline code (config.py + analysis_io.py).  Read-only; same logic as
    ``load_canada`` but with an explicit output root.
    """
    root, pipeline_dir = Path(root).resolve(), Path(pipeline_dir).resolve()
    os.environ["PROVENANCE_OUTPUT_DIR"] = str(root)
    sys.path.insert(0, str(pipeline_dir))
    import analysis_io
    if backend == "nomic_embed_text":
        source, official = analysis_io.nomic_core_embeddings(topic, root)
        blocks = None
    else:
        blocks = analysis_io.embedding_blocks(topic, backend, root)
        source, official = blocks["records"], blocks["summary"]
    path = root / topic / f"embeddings_{backend}.npz"
    targets = {"official": official}
    descriptions = {"official": {"kind": "official"}}
    if n_random and blocks is not None:
        offset = 0
        for index, draw in enumerate(blocks["null_draws"][:n_random]):
            name = f"exact_budget_random_{index:04d}"
            targets[name] = blocks["null_embeddings"][offset:offset + len(draw)]
            descriptions[name] = {"kind": "exact_word_budget_random_windows",
                                  "words": sum(item["word_count"] for item in draw),
                                  "warning": "Windows may be sentence fragments; geometric reference only"}
            offset += len(draw)
    elif n_random:
        raise ValueError("Nomic archive contains no random-window embeddings; use --random-references 0")
    dependencies = [path, root / topic / "clean.parquet",
                    pipeline_dir / "config.py", pipeline_dir / "analysis_io.py"]
    if blocks is not None:
        dependencies.extend([root / topic / "complete_sentence_candidates.parquet",
                             root / topic / "exact_random_windows.json"])
    return source, targets, descriptions, dependencies, f"{root.parent.name}_{topic}"


def load_dataset(directory, backend):
    directory = Path(directory).resolve()
    path = directory / "embeddings" / f"{backend}.npz"
    with np.load(path, allow_pickle=False) as saved:
        source, summary = np.asarray(saved["records"]), np.asarray(saved["summary"])
    records_file, summary_file = directory / "records.jsonl", directory / "summary.json"
    # JSON strings can legally contain U+2028/U+0085. str.splitlines() treats
    # those as row breaks even when they occur inside a valid JSON string.
    records = [json.loads(line) for line in records_file.read_text().split("\n") if line.strip()]
    summary_text = json.loads(summary_file.read_text())
    if len(records) != len(source) or len(summary_text["sentences"]) != len(summary):
        raise ValueError("Embedding row counts differ from records.jsonl/summary.json")
    if len({str(record["respondent_id"]) for record in records}) != len(records):
        raise ValueError("Duplicate respondent IDs in corpus; one source unit per respondent/question required")
    paths = [path, records_file, summary_file]
    metadata = {"official": {"kind": "official"}}
    for candidate in [directory / "dataset.json", path.with_suffix(".json")]:
        if candidate.exists():
            paths.append(candidate)
    if path.with_suffix(".json").exists():
        embedding_metadata = json.loads(path.with_suffix(".json").read_text())
        identity = embedding_metadata.get("identity")
        if identity:
            actual_digest = text_fingerprint([row["text"] for row in records] + summary_text["sentences"])
            if embedding_metadata.get("text_digest") != actual_digest:
                raise ValueError("Dataset text changed after embedding (text fingerprint mismatch)")
            metadata["official"]["embedding_identity"] = identity
    return source, {"official": summary}, metadata, paths, directory.name


def flat_rows(result):
    rows = []
    for run in result["runs"]:
        for name, scores in run["targets"].items():
            row = {"dataset": result["dataset"], "backend": result["backend"],
                   "pca_dimensions": run["pca_dimensions"], "target": name,
                   "target_kind": result["target_metadata"][name].get("kind", "additional"),
                   "n_records": result["n_records"], "n_summary_units": scores["n_summary_units"],
                   "pca_explained_variance": run["pca_explained_variance"],
                   "wasserstein_1": scores["wasserstein_1"], "wasserstein_2": scores["wasserstein_2"]}
            for multiplier, mmd in scores["mmd2"].items():
                row[f"mmd2_bandwidth_x{multiplier}"] = mmd["value"]
            rows.append(row)
    return rows


def comparisons(result):
    """Descriptive ranking only: targets, not six summary sentences, are units."""
    outputs = []
    for run in result["runs"]:
        names = list(run["targets"])
        if len(names) < 3:
            outputs.append({"pca_dimensions": run["pca_dimensions"],
                            "ranking_agreement": "Requires at least three summary targets"})
            continue
        series = {"wasserstein_1": [], "wasserstein_2": []}
        for multiplier in next(iter(run["targets"].values()))["mmd2"]:
            series[f"mmd2_x{multiplier}"] = []
        for scores in run["targets"].values():
            for metric in ("wasserstein_1", "wasserstein_2"):
                series[metric].append(scores[metric])
            for multiplier, value in scores["mmd2"].items():
                series[f"mmd2_x{multiplier}"].append(value["value"])
        coefficients = {}
        baseline = series["wasserstein_2"]
        for metric, values in series.items():
            coefficient = (float(spearmanr(baseline, values).statistic)
                           if np.ptp(baseline) > 0 and np.ptp(values) > 0 else None)
            coefficients[metric] = coefficient
        outputs.append({"pca_dimensions": run["pca_dimensions"], "n_targets": len(names),
                        "target_names": names, "spearman_rank_vs_w2": coefficients,
                        "interpretation": "Descriptive agreement among supplied targets; no p-values"})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dataset", type=Path, help="Prepared dataset directory")
    source_group.add_argument("--canada", choices=["education", "trust"])
    source_group.add_argument("--original-root", type=Path,
                              help="Analysis directory of a re-run of the original pipeline "
                                   "(e.g. analyses/scotland_analysis/analysis); needs --topic")
    parser.add_argument("--original-pipeline", type=Path,
                        help="Pipeline directory that produced --original-root "
                             "(e.g. analyses/scotland_analysis/pipeline)")
    parser.add_argument("--topic", help="Topic directory under --original-root (e.g. q7)")
    parser.add_argument("--backend", default="openai")
    parser.add_argument("--target", action="append", default=[], metavar="NAME=NPZ",
                        help="Additional summary embeddings from identical embedding backend")
    parser.add_argument("--target-manifest", type=Path,
                        help='JSON {"targets":[{"name":"...","path":"...","kind":"generated"}]}')
    parser.add_argument("--dims", default="20,50,100")
    parser.add_argument("--bandwidths", default="0.5,1,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", type=Path, help="Optional frozen group labels.npy, one per participant row")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--random-references", type=int, default=0,
                        help="Canada only: first N archived exact-word-budget null draws")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.random_references < 0:
        parser.error("--random-references cannot be negative")
    if args.dataset and args.random_references:
        parser.error("--random-references is supported only by --canada")
    if args.original_root and not (args.original_pipeline and args.topic):
        parser.error("--original-root requires --original-pipeline and --topic")
    if not args.original_root and (args.original_pipeline or args.topic):
        parser.error("--original-pipeline/--topic are only valid with --original-root")
    if args.original_root:
        source, targets, metadata, paths, name = load_original(
            args.original_root, args.original_pipeline, args.topic,
            args.backend, args.random_references)
    elif args.canada:
        source, targets, metadata, paths, name = load_canada(args.canada, args.backend, args.random_references)
    else:
        source, targets, metadata, paths, name = load_dataset(args.dataset, args.backend)
    entries = []
    if args.target_manifest:
        manifest = json.loads(args.target_manifest.read_text())
        paths.append(args.target_manifest)
        for item in manifest["targets"]:
            entry = dict(item)
            entry["path"] = str((args.target_manifest.parent / item["path"]).resolve())
            entries.append(entry)
    for argument in args.target:
        target_name, path = argument.split("=", 1)
        entries.append({"name": target_name, "path": path, "kind": "additional"})
    for item in entries:
        target_name, path = item["name"], Path(item["path"]).expanduser().resolve()
        if not target_name or target_name in targets:
            raise ValueError(f"Duplicate/empty target name: {target_name}")
        if path.with_suffix(".json").exists():
            target_sidecar = json.loads(path.with_suffix(".json").read_text())
            source_identity = metadata["official"].get("embedding_identity")
            target_identity = target_sidecar.get("identity")
            if source_identity and target_identity and source_identity != target_identity:
                raise ValueError(f"{target_name}: embedding model or preprocessing identity differs from source")
            paths.append(path.with_suffix(".json"))
        targets[target_name] = load_target(path)
        metadata[target_name] = {k: v for k, v in item.items() if k not in {"path", "name"}}
        paths.append(path)
    labels = None
    if args.labels:
        labels = np.load(args.labels, allow_pickle=False)
        paths.append(args.labels)
    result = evaluate_targets(source, targets, dims=[int(v) for v in args.dims.split(",")],
                              bandwidth_multipliers=[float(v) for v in args.bandwidths.split(",")],
                              seed=args.seed, block_size=args.block_size, labels=labels)
    result.update({"dataset": name, "backend": args.backend, "target_metadata": metadata,
                   "inputs": [{"path": str(Path(path).resolve()), "sha256": sha256(path)} for path in paths],
                   "command": sys.argv, "rank_agreement": comparisons(result),
                   "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "software": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "POT")},
                   "python_version": sys.version,
                   "code_sha256": {name: sha256(Path(__file__).parent / name) for name in ("run.py", "metrics.py")}})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rows = flat_rows(result)
    with (args.output / "distances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} target/PCA comparisons to {args.output}")


if __name__ == "__main__":
    main()
