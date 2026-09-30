"""Inspect the launcher command without importing vLLM or touching a GPU."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SERVE = Path(__file__).resolve().parents[1] / "serve.sh"


class ServeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.capture = self.root / "arguments.json"
        self.python = self.root / "capture-python"
        self.python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['CAPTURE_ARGUMENTS'], 'w') as f:\n"
            "    json.dump({'args': sys.argv[1:], "
            "'gpus': os.environ['CUDA_VISIBLE_DEVICES']}, f)\n",
            encoding="utf-8")
        self.python.chmod(0o755)

    def run_serve(self, checkpoint, **overrides):
        env = os.environ.copy()
        for key in ("CKPT", "NAME", "GPUS", "TP", "PY", "PORT", "MAX_LEN",
                    "MEM_UTIL", "MAX_IMGS", "QUANT", "EAGER"):
            env.pop(key, None)
        env.update(CKPT=checkpoint, GPUS="0", TP="1", PY=str(self.python),
                   CAPTURE_ARGUMENTS=str(self.capture))
        env.update(overrides)
        return subprocess.run(["bash", str(SERVE)], env=env, cwd=self.root,
                              text=True, capture_output=True, timeout=10)

    def assert_launch(self, result, checkpoint, name):
        self.assertEqual(result.returncode, 0, result.stderr)
        launch = json.loads(self.capture.read_text(encoding="utf-8"))
        args = launch["args"]
        self.assertEqual(args[:2], ["-m", "vllm.entrypoints.openai.api_server"])
        self.assertEqual(args[args.index("--model") + 1], checkpoint)
        self.assertEqual(args[args.index("--served-model-name") + 1], name)
        return launch

    def test_hub_id_is_accepted_and_keeps_its_full_served_name(self):
        for checkpoint in ("Qwen/Qwen-Image-2.1-PE-T2I", "Qwen/Qwen-Image-2.1-PE-I2I"):
            with self.subTest(checkpoint=checkpoint):
                self.assert_launch(self.run_serve(checkpoint), checkpoint, checkpoint)

    def test_local_directory_keeps_basename_and_spaces(self):
        checkpoint = self.root / "local checkpoint"
        checkpoint.mkdir()
        self.assert_launch(self.run_serve(str(checkpoint)), str(checkpoint), checkpoint.name)

    def test_explicit_served_name_and_engine_options_are_preserved(self):
        checkpoint = "Qwen/Qwen-Image-2.1-PE-T2I"
        launch = self.assert_launch(self.run_serve(
            checkpoint, NAME="custom-name", GPUS="2,3", TP="2", QUANT="fp8", EAGER="1"),
            checkpoint, "custom-name")
        self.assertEqual(launch["gpus"], "2,3")
        args = launch["args"]
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(args[args.index("--quantization") + 1], "fp8")
        self.assertIn("--enforce-eager", args)

    def test_checkpoint_is_still_required(self):
        result = self.run_serve("")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CKPT", result.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
