"""System-prompt resolution without model downloads or a GPU."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from huggingface_hub.errors import (HFValidationError, HfHubHTTPError,
                                    LocalEntryNotFoundError)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pe_core as core


class SystemPromptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cached = self.root / "cached_prompt.txt"
        self.cached.write_text("  checkpoint prompt 中文\n", encoding="utf-8")
        download = patch("huggingface_hub.hf_hub_download", return_value=str(self.cached))
        self.download = download.start()
        self.addCleanup(download.stop)

    def test_explicit_file_takes_precedence_over_checkpoint(self):
        self.assertEqual(
            core.load_system_prompt(str(self.cached), "Qwen/Qwen-Image-2.1-PE-T2I"),
            "checkpoint prompt 中文")
        self.download.assert_not_called()

    def test_missing_explicit_file_does_not_fall_back(self):
        with self.assertRaisesRegex(SystemExit, "--system-prompt.*not a file"):
            core.load_system_prompt(str(self.root / "missing.txt"), "owner/model")
        self.download.assert_not_called()

    def test_local_checkpoint_prompt(self):
        (self.root / "system_prompt.txt").write_text("local prompt\n", encoding="utf-8")
        self.assertEqual(core.load_system_prompt(None, str(self.root)), "local prompt")
        self.download.assert_not_called()

    def test_local_checkpoint_missing_prompt_stays_local(self):
        with self.assertRaisesRegex(SystemExit, "--system-prompt"):
            core.load_system_prompt(None, str(self.root))
        self.download.assert_not_called()

    def test_missing_local_paths_do_not_trigger_hub_downloads(self):
        for checkpoint in (str(self.root / "missing"), "./missing", "../missing"):
            with self.subTest(checkpoint=checkpoint):
                with self.assertRaisesRegex(SystemExit, "--system-prompt"):
                    core.load_system_prompt(None, checkpoint)
        self.download.assert_not_called()

    def test_hub_checkpoint_downloads_only_its_own_prompt(self):
        for checkpoint in ("Qwen/Qwen-Image-2.1-PE-T2I",
                           "Qwen/Qwen-Image-2.1-PE-I2I", "owner/custom-checkpoint"):
            with self.subTest(checkpoint=checkpoint):
                self.download.reset_mock()
                self.assertEqual(core.load_system_prompt(None, checkpoint),
                                 "checkpoint prompt 中文")
                self.download.assert_called_once_with(
                    repo_id=checkpoint, filename="system_prompt.txt")

    def test_hub_failures_explain_explicit_override(self):
        response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))
        for error in (HfHubHTTPError("file unavailable", response=response),
                      LocalEntryNotFoundError("not cached while offline"),
                      HFValidationError("invalid repository ID"),
                      OSError("cache unavailable")):
            with self.subTest(error=type(error).__name__):
                self.download.side_effect = error
                with self.assertRaisesRegex(SystemExit, "owner/model.*--system-prompt"):
                    core.load_system_prompt(None, "owner/model")

    def test_missing_checkpoint_requires_explicit_prompt(self):
        with self.assertRaisesRegex(SystemExit, "--system-prompt"):
            core.load_system_prompt(None, None)
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
