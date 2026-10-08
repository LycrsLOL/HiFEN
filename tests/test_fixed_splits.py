"""Run with: python -m unittest discover -s tests -p test_fixed_splits.py"""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hifen.data.fixed_splits import fixed_split_directory, load_fixed_splits


class FixedSplitsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.graph_dir = self.root / "graphs"
        self.graph_dir.mkdir()
        self.names = {"train": ["NP_001.1.pt", "P12345_1abc.pt"], "val": ["WP_002.1.pt"], "test": ["XP_003.1.pt"]}
        self.summary = {"strategy": "fixed_s2", "manifest_format": "graph_basename", "counts": {}, "manifest_sha256": {}, "total_records": 4}
        for split, names in self.names.items():
            path = self.root / f"{split}_graphs.txt"
            path.write_text("".join(f"{name}\n" for name in names))
            self.summary["counts"][split] = len(names)
            self.summary["manifest_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
            for name in names:
                (self.graph_dir / name).touch()
        self.save_summary()
        self.config = {"train": {"fixed_split_dir": str(self.root)}, "storage": {"graph_dir": str(self.graph_dir), "split_dir": str(self.root / "runtime")}}

    def save_summary(self):
        (self.root / "split_summary.json").write_text(json.dumps(self.summary))

    def test_exact_assignments_keep_full_refseq_ids_and_ignore_extra_graphs(self):
        (self.graph_dir / "excluded.pt").touch()
        paths, summary = load_fixed_splits(self.config)
        self.assertEqual({split: [Path(p).name for p in values] for split, values in paths.items()}, self.names)
        self.assertEqual(summary["counts"], {"train": 2, "val": 1, "test": 1})

    def test_profile_uses_its_own_graph_cache(self):
        alternate = self.root / "without_completion"
        alternate.mkdir()
        for names in self.names.values():
            for name in names:
                (alternate / name).touch()
        self.config["storage"]["graph_dir"] = str(alternate)
        paths, _ = load_fixed_splits(self.config)
        self.assertTrue(all(Path(p).parent == alternate for values in paths.values() for p in values))

    def test_missing_graph_is_fatal_and_does_not_modify_manifests(self):
        before = (self.root / "train_graphs.txt").read_bytes()
        (self.graph_dir / self.names["train"][0]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing 1 prepared graphs"):
            load_fixed_splits(self.config)
        self.assertEqual((self.root / "train_graphs.txt").read_bytes(), before)

    def test_changed_manifest_is_rejected(self):
        (self.root / "train_graphs.txt").write_text("different.pt\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_fixed_splits(self.config)

    def test_duplicate_across_splits_is_rejected_even_with_updated_hash(self):
        path = self.root / "test_graphs.txt"
        path.write_text(self.names["val"][0] + "\n")
        self.summary["manifest_sha256"]["test"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_summary()
        with self.assertRaisesRegex(ValueError, "cross-split"):
            load_fixed_splits(self.config)

    def test_force_resplit_conflicts_with_fixed_cohort(self):
        self.config["train"]["force_resplit"] = True
        with self.assertRaisesRegex(ValueError, "force_resplit conflicts"):
            load_fixed_splits(self.config)

    def test_limit_preserves_assignments_and_only_checks_selected_graphs(self):
        (self.graph_dir / self.names["train"][1]).unlink()
        paths, _ = load_fixed_splits(self.config, limit=1)
        self.assertEqual([Path(p).name for p in paths["train"]], [self.names["train"][0]])
        self.assertEqual({name: len(values) for name, values in paths.items()}, {"train": 1, "val": 1, "test": 1})

    def test_missing_summary_is_fatal(self):
        (self.root / "split_summary.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing fixed split summary"):
            load_fixed_splits(self.config)

    def test_relative_manifest_dir_is_resolved_from_config_location(self):
        self.config["_config_dir"] = str(self.root / "configs")
        self.config["train"]["fixed_split_dir"] = "."
        self.assertEqual(fixed_split_directory(self.config), self.root)

    def test_training_loader_prioritizes_s2_over_stale_runtime_splits(self):
        # Execute the actual loader function in isolation; neural dependencies
        # are not needed to verify partition routing and runtime persistence.
        source = Path(__file__).resolve().parents[1] / "hifen/training/trainer.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_or_create_splits")
        namespace = {
            "Path": Path, "fixed_split_directory": fixed_split_directory, "load_fixed_splits": load_fixed_splits,
            "_read_lines": lambda p: p.read_text().split(),
            "_write_lines": lambda p, values: p.write_text("".join(f"{value}\n" for value in values)),
            "_load_json": lambda p: json.loads(p.read_text()),
            "write_json": lambda p, value: p.write_text(json.dumps(value)),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        runtime = Path(self.config["storage"]["split_dir"])
        runtime.mkdir()
        (runtime / "train_graphs.txt").write_text("stale.pt\n")
        paths = namespace["load_or_create_splits"](self.config)
        self.assertEqual((runtime / "train_graphs.txt").read_text().splitlines(), paths["train"])
        self.assertEqual(json.loads((runtime / "split_summary.json").read_text())["counts"], self.summary["counts"])
        self.assertEqual((self.root / "train_graphs.txt").read_text().splitlines(), self.names["train"])


if __name__ == "__main__":
    unittest.main()
