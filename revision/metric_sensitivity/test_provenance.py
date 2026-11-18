"""Reject stale or mixed embedding artifacts before producing a comparison."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run


class ProvenanceTests(unittest.TestCase):
    def make_dataset(self, root):
        dataset = root / "dataset"
        (dataset / "embeddings").mkdir(parents=True)
        records = [{"respondent_id": str(i), "text": f"Submission {i}"} for i in range(4)]
        sentences = ["Official synthesis."]
        (dataset / "records.jsonl").write_text("\n".join(json.dumps(v) for v in records))
        (dataset / "summary.json").write_text(json.dumps({"sentences": sentences}))
        np.savez(dataset / "embeddings" / "example.npz", records=np.arange(8).reshape(4, 2), summary=[[0., 1.]])
        sidecar = {"identity": {"model": "one", "policy": "full"},
                   "text_digest": run.text_fingerprint([row["text"] for row in records] + sentences)}
        (dataset / "embeddings" / "example.json").write_text(json.dumps(sidecar))
        return dataset

    def test_new_text_invalidates_existing_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary))
            run.load_dataset(dataset, "example")
            path = dataset / "records.jsonl"
            path.write_text(path.read_text().replace("Submission 0", "Changed statement"))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run.load_dataset(dataset, "example")

    def test_unicode_line_separators_inside_json_string_are_not_record_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary))
            path = dataset / "records.jsonl"
            records = [json.loads(line) for line in path.read_text().split("\n")]
            records[0]["text"] = "Submission\u20280\u0085continued"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records))
            summary = json.loads((dataset / "summary.json").read_text())["sentences"]
            sidecar_path = dataset / "embeddings" / "example.json"
            metadata = json.loads(sidecar_path.read_text())
            metadata["text_digest"] = run.text_fingerprint([row["text"] for row in records] + summary)
            sidecar_path.write_text(json.dumps(metadata))
            values, *_ = run.load_dataset(dataset, "example")
            self.assertEqual(len(values), 4)

    def test_target_backend_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.make_dataset(root)
            target = root / "target.npz"
            np.savez(target, summary=[[0., 1.]])
            target.with_suffix(".json").write_text(json.dumps({"identity": {"model": "two", "policy": "full"}}))
            arguments = ["run.py", "--dataset", str(dataset), "--backend", "example",
                         "--target", f"alternative={target}", "--dims", "1", "--output", str(root / "out")]
            with patch("sys.argv", arguments), self.assertRaisesRegex(ValueError, "identity differs"):
                run.main()
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
