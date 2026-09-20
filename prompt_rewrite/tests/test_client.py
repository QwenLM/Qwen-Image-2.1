"""Exercise the CLI through a mocked streaming endpoint; no server required."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from huggingface_hub.errors import LocalEntryNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.prompt_path = self.root / "system_prompt.txt"
        self.prompt_path.write_text("Checkpoint-specific prompt", encoding="utf-8")
        patches = contextlib.ExitStack()
        self.addCleanup(patches.close)
        self.download = patches.enter_context(patch(
            "huggingface_hub.hf_hub_download", return_value=str(self.prompt_path)))
        self.endpoint = Mock()
        patches.enter_context(patch.object(client, "OpenAI", return_value=self.endpoint))
        answer = json.dumps({"rewritten_prompt": "A corgi playing guitar.",
                             "wh_ratio": "16:9"})
        self.endpoint.chat.completions.create.return_value = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning_content="Consider the composition.", content=None))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning_content=None, content=answer))]),
        ]

    def run_client(self, *args):
        output = io.StringIO()
        with patch.object(sys, "argv", ["client.py", *args]), contextlib.redirect_stdout(output):
            status = client.main()
        return status, output.getvalue()

    def test_documented_hub_command_resolves_prompt_and_emits_record(self):
        model = "Qwen/Qwen-Image-2.1-PE-T2I"
        status, output = self.run_client("--task", "t2i", "--model", model,
                                         "a corgi playing guitar")
        self.assertEqual(status, 0)
        record = json.loads(output)
        self.assertEqual(record["positive_prompt"], "A corgi playing guitar.")
        self.assertEqual(record["wh_ratio"], "16:9")
        self.assertTrue(record["parse_ok"])
        self.download.assert_called_once_with(repo_id=model, filename="system_prompt.txt")
        request = self.endpoint.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], model)
        self.assertEqual(request["messages"][0], {
            "role": "system", "content": "Checkpoint-specific prompt"})

    def test_served_alias_with_explicit_prompt_needs_no_hub(self):
        status, _ = self.run_client("--task", "t2i", "--model", "my-alias",
                                   "--system-prompt", str(self.prompt_path), "a corgi")
        self.assertEqual(status, 0)
        self.download.assert_not_called()
        request = self.endpoint.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "my-alias")

    def test_edit_uses_its_own_prompt_and_preserves_image_order(self):
        model = "Qwen/Qwen-Image-2.1-PE-I2I"
        images = [self.root / "first.png", self.root / "second.png"]
        for image in images:
            image.touch()
        answer = json.dumps({"rewritten_prompt": "Place the first subject in the second scene.",
                             "wh_ratio": "", "ratio_follow": "<image2>"})
        self.endpoint.chat.completions.create.return_value = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=answer))])]
        with patch.object(client, "image_to_data_uri", side_effect=["data:first", "data:second"]):
            status, output = self.run_client(
                "--task", "edit", "--model", model, "--image", str(images[0]),
                "--image", str(images[1]), "put the first subject in the second scene")
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output)["ratio_follow"], "<image2>")
        self.download.assert_called_once_with(repo_id=model, filename="system_prompt.txt")
        content = self.endpoint.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertEqual([part["image_url"]["url"] for part in content[:2]],
                         ["data:first", "data:second"])

    def test_missing_hub_prompt_fails_before_inference(self):
        self.download.side_effect = LocalEntryNotFoundError("not cached")
        with self.assertRaisesRegex(SystemExit, "--system-prompt"):
            self.run_client("--task", "t2i", "--model", "owner/model", "a corgi")
        self.endpoint.chat.completions.create.assert_not_called()

    def test_batch_uses_the_same_hub_prompt_and_output_contract(self):
        input_path = self.root / "input.jsonl"
        input_path.write_text('{"id":"sample","prompt":"a corgi"}\n', encoding="utf-8")
        output_path = self.root / "output.jsonl"
        status, _ = self.run_client(
            "--task", "t2i", "--model", "Qwen/Qwen-Image-2.1-PE-T2I",
            "--input", str(input_path), "--output", str(output_path))
        self.assertEqual(status, 0)
        record = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(record["id"], "sample")
        self.assertEqual(record["positive_prompt"], "A corgi playing guitar.")
        self.assertTrue(record["parse_ok"])


if __name__ == "__main__":
    unittest.main()
