import json
from pathlib import Path
import tempfile
import unittest

from folding_workflow.baseline import freeze, hash_path, require_mutable, verify


class BaselineTest(unittest.TestCase):
    def test_freeze_guards_raw_tree_and_verifies_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            artifact = root / "artifact"
            raw.mkdir()
            artifact.mkdir()
            (raw / "data.bin").write_bytes(b"raw")
            (artifact / "model.bin").write_bytes(b"model")
            manifest = root / "baseline/manifest.json"

            freeze("test", raw, manifest, {"raw": raw, "model": artifact}, {"kind": "test"})
            ok, results = verify(manifest)
            self.assertTrue(ok, results)
            with self.assertRaisesRegex(SystemExit, "Dataset is frozen"):
                require_mutable(raw, "change data")

            (artifact / "model.bin").write_bytes(b"changed")
            ok, results = verify(manifest)
            self.assertFalse(ok)
            self.assertFalse(next(item for item in results if item["artifact"] == "model")["ok"])

    def test_marker_and_cache_files_do_not_change_tree_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").write_text("stable", encoding="utf-8")
            before = hash_path(root)
            (root / ".openarm-fold-frozen.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "module.pyc").write_bytes(b"cache")
            self.assertEqual(hash_path(root), before)


if __name__ == "__main__":
    unittest.main()
