"""Empirical distribution distances with shared, participant-fitted geometry."""
from __future__ import annotations

import numpy as np
import ot
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA


def validate_embeddings(values, name="embeddings"):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name}: expected nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: found non-finite values")
    return array


def transport_distances(source, target):
    """W1 uses Euclidean cost; W2 is sqrt of optimal squared Euclidean cost.

    Both solve balanced exact OT with uniform empirical observation weights.
    The W1 and W2 plans are optimized independently, not reused.
    """
    source = validate_embeddings(source, "source")
    target = validate_embeddings(target, "target")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target dimensions differ")
    squared = np.maximum(cdist(source, target, metric="sqeuclidean"), 0)
    a, b = np.full(len(source), 1 / len(source)), np.full(len(target), 1 / len(target))
    w1 = float(ot.emd2(a, b, np.sqrt(squared), numItermax=1000000))
    w2 = float(np.sqrt(max(0.0, ot.emd2(a, b, squared, numItermax=1000000))))
    if w1 > w2 + 1e-7 * max(1, w2):
        raise RuntimeError("OT solver violated W1 <= W2; check convergence")
    return {"wasserstein_1": w1, "wasserstein_2": w2}


def source_bandwidth(source, seed=42, sample_size=1000):
    """Median positive pair distance, using source rows only.

    With >1000 rows a deterministic source subsample controls quadratic work.
    The bandwidth must remain fixed across every target for this source/PCA.
    """
    source = validate_embeddings(source, "source")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(source), min(sample_size, len(source)), replace=False))
    distances = pdist(source[indices], metric="euclidean")
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError("Cannot estimate Gaussian bandwidth: all sampled source rows coincide")
    return float(np.median(positive)), indices


def gaussian_kernel_mean(left, right, bandwidth, block_size=512):
    """Mean over all ordered row pairs, including diagonal when left == right."""
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Gaussian bandwidth must be positive and finite")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    total = 0.0
    denominator = 2 * bandwidth * bandwidth
    for start in range(0, len(left), block_size):
        squared = cdist(left[start:start + block_size], right, metric="sqeuclidean")
        total += np.exp(-np.maximum(squared, 0) / denominator).sum(dtype=np.float64)
    return float(total / (len(left) * len(right)))


def empirical_mmd2(source, target, bandwidth, source_kernel_mean=None, block_size=512):
    """Squared RKHS distance of two empirical distributions (V statistic).

    Includes self-pairs. This is a nonnegative descriptive distance; it is not
    the off-diagonal unbiased estimator and has no associated p-value here.
    """
    source = validate_embeddings(source, "source")
    target = validate_embeddings(target, "target")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target dimensions differ")
    xx = (gaussian_kernel_mean(source, source, bandwidth, block_size)
          if source_kernel_mean is None else source_kernel_mean)
    yy = gaussian_kernel_mean(target, target, bandwidth, block_size)
    xy = gaussian_kernel_mean(source, target, bandwidth, block_size)
    value = float(xx + yy - 2 * xy)
    if value < -1e-9:
        raise RuntimeError(f"Materially negative empirical MMD² ({value})")
    return max(0.0, value)


def evaluate_targets(records, targets, dims=(20, 50, 100),
                     bandwidth_multipliers=(0.5, 1.0, 2.0), seed=42, block_size=512,
                     labels=None):
    """Evaluate all targets in one shared participant-fitted space per dimension.

    No target can change the source coordinates, PCA basis or kernel bandwidth.
    Input arrays retain their original scaling; no silent row normalization.
    """
    records = validate_embeddings(records, "records")
    targets = {str(k): validate_embeddings(v, str(k)) for k, v in targets.items()}
    if not targets:
        raise ValueError("At least one summary target is required")
    if any(v.shape[1] != records.shape[1] for v in targets.values()):
        raise ValueError("Target and participant embedding dimensions differ")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(records):
            raise ValueError("Group labels must have exactly one value per participant row")
    if len(records) < 2:
        raise ValueError("At least two source records are required for participant-fitted PCA")
    dimensions = list(dict.fromkeys(int(v) for v in dims))
    if any(v < 1 for v in dimensions):
        raise ValueError("PCA dimensions must be positive integers")
    multipliers = list(dict.fromkeys(float(v) for v in bandwidth_multipliers))
    if any(not np.isfinite(v) or v <= 0 for v in multipliers):
        raise ValueError("Bandwidth multipliers must be positive finite numbers")
    result = {"geometry": "PCA fit on participant embeddings only; uniform empirical masses",
              "normalization": "none; retain input embedding scale",
              "mmd_estimator": "biased empirical MMD squared; includes self-pairs",
              "seed": seed, "n_records": len(records), "embedding_dimensions": records.shape[1],
              "runs": [], "skipped_dimensions": []}
    rank_bound = min(len(records) - 1, records.shape[1])
    valid_dimensions = [d for d in dimensions if d <= rank_bound]
    result["skipped_dimensions"] = [d for d in dimensions if d > rank_bound]
    if not valid_dimensions:
        raise ValueError("Every requested PCA dimension exceeded the participant matrix rank bound")
    # Fit once: prefixes of a full-SVD PCA give the same solution at lower
    # dimensions and avoid repeating the expensive participant decomposition.
    pca = PCA(n_components=max(valid_dimensions), svd_solver="full")
    source_all = pca.fit_transform(records)
    # Direct centered contraction is equivalent to PCA.transform; avoids small-
    # target BLAS floating-point flag warnings on some Apple Accelerate builds.
    target_all = {name: np.einsum("ij,kj->ik", target - pca.mean_, pca.components_, optimize=False)
                  for name, target in targets.items()}
    for dimension in valid_dimensions:
        source = source_all[:, :dimension]
        bandwidth, bandwidth_indices = source_bandwidth(source, seed=seed)
        source_terms = {m: gaussian_kernel_mean(source, source, bandwidth * m, block_size)
                        for m in multipliers}
        run = {"pca_dimensions": dimension,
               "pca_explained_variance": float(pca.explained_variance_ratio_[:dimension].sum()),
               "source_bandwidth": bandwidth,
               "bandwidth_source_row_indices": bandwidth_indices.tolist(),
               "targets": {}}
        for name, target in targets.items():
            projected = target_all[name][:, :dimension]
            measures = transport_distances(source, projected)
            measures.update({"n_summary_units": len(target), "mmd2": {}})
            for multiplier in multipliers:
                score = empirical_mmd2(source, projected, bandwidth * multiplier,
                                       source_kernel_mean=source_terms[multiplier], block_size=block_size)
                measures["mmd2"][str(multiplier)] = {
                    "bandwidth": bandwidth * multiplier, "value": score,
                    "mmd": float(np.sqrt(score))}
            run["targets"][name] = measures
        if labels is not None:
            run["groups"] = {}
            for label in np.unique(labels):
                subgroup = source[labels == label]
                group_terms = {m: gaussian_kernel_mean(subgroup, subgroup, bandwidth * m, block_size)
                               for m in multipliers}
                group = {"n_records": len(subgroup), "targets": {}}
                for name, projected_all in target_all.items():
                    projected = projected_all[:, :dimension]
                    scores = transport_distances(subgroup, projected)
                    scores["mmd2"] = {
                        str(m): empirical_mmd2(subgroup, projected, bandwidth * m,
                                              group_terms[m], block_size)
                        for m in multipliers}
                    group["targets"][name] = scores
                run["groups"][str(label)] = group
        result["runs"].append(run)
    if not result["runs"]:
        raise ValueError("Every requested PCA dimension exceeded the participant matrix rank bound")
    return result
