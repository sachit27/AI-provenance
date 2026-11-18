"""Meaningful metric invariants and known mathematical examples; no model calls."""
import unittest

import numpy as np

from metrics import (empirical_mmd2, evaluate_targets, gaussian_kernel_mean,
                     source_bandwidth, transport_distances)


class DistanceTests(unittest.TestCase):
    def test_wasserstein_known_one_dimensional_solution(self):
        result = transport_distances(np.array([[0.], [2.]]), np.array([[0.]]))
        self.assertAlmostEqual(result["wasserstein_1"], 1.0)
        self.assertAlmostEqual(result["wasserstein_2"], np.sqrt(2))

    def test_identical_empirical_distributions_have_zero_distance(self):
        source = np.array([[0., 1.], [1., 2.], [3., 4.]])
        result = transport_distances(source, source)
        self.assertAlmostEqual(result["wasserstein_1"], 0)
        self.assertAlmostEqual(result["wasserstein_2"], 0)
        self.assertAlmostEqual(empirical_mmd2(source, source, 1), 0)

    def test_mmd_closed_form_and_cached_kernel_agree(self):
        source, target = np.array([[0.], [2.]]), np.array([[0.]])
        # k00=k22=1, k02=exp(-2). MMD²=(1-exp(-2))/2.
        expected = (1 - np.exp(-2)) / 2
        self.assertAlmostEqual(empirical_mmd2(source, target, 1), expected)
        cached = gaussian_kernel_mean(source, source, 1, block_size=1)
        self.assertAlmostEqual(empirical_mmd2(source, target, 1, cached, block_size=1), expected)

    def test_distribution_preserving_duplication(self):
        source, target = np.array([[0.], [2.]]), np.array([[1.], [3.]])
        duplicated = np.repeat(source, 3, axis=0)
        original, repeated = transport_distances(source, target), transport_distances(duplicated, target)
        for metric in original:
            self.assertAlmostEqual(original[metric], repeated[metric])
        self.assertAlmostEqual(empirical_mmd2(source, target, 0.7),
                               empirical_mmd2(duplicated, target, 0.7))

    def test_source_only_geometry_independent_of_new_target(self):
        source = np.random.default_rng(4).normal(size=(30, 4))
        official = source[[0, 4]]
        first = evaluate_targets(source, {"official": official}, dims=[2])
        second = evaluate_targets(source, {"official": official, "extreme": np.full((3, 4), 1e6)}, dims=[2])
        a, b = first["runs"][0], second["runs"][0]
        self.assertEqual(a["source_bandwidth"], b["source_bandwidth"])
        self.assertEqual(a["targets"]["official"], b["targets"]["official"])

    def test_pca_prefix_matches_standalone_fit(self):
        source = np.random.default_rng(9).normal(size=(30, 5))
        target = source[[2, 5]]
        separate = evaluate_targets(source, {"a": target}, dims=[2])["runs"][0]
        joint = evaluate_targets(source, {"a": target}, dims=[2, 4])["runs"][0]
        self.assertEqual(separate["targets"], joint["targets"])

    def test_bandwidth_is_source_determined_and_failures_are_explicit(self):
        value, indices = source_bandwidth(np.array([[0.], [1.], [2.]]))
        self.assertEqual(value, 1)
        self.assertEqual(len(indices), 3)
        with self.assertRaises(ValueError):
            source_bandwidth(np.ones((5, 2)))
        with self.assertRaises(ValueError):
            transport_distances([[float("nan")]], [[1.]])
        with self.assertRaises(ValueError):
            evaluate_targets(np.ones((2, 3)), {"a": np.ones((1, 3))}, dims=[3])

    def test_group_diagnostics_use_shared_global_geometry(self):
        source = np.random.default_rng(11).normal(size=(20, 3))
        labels = np.repeat([0, 1], 10)
        result = evaluate_targets(source, {"a": source[[0, 2]]}, dims=[2], labels=labels)
        run = result["runs"][0]
        self.assertEqual(run["groups"]["0"]["n_records"], 10)
        self.assertEqual(run["groups"]["1"]["n_records"], 10)
        self.assertGreaterEqual(run["groups"]["0"]["targets"]["a"]["mmd2"]["1.0"], 0)
        with self.assertRaises(ValueError):
            evaluate_targets(source, {"a": source[[0]]}, dims=[2], labels=[0])


if __name__ == "__main__":
    unittest.main()
