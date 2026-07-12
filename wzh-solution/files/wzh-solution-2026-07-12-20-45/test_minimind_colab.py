import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNNER_PATH = Path(__file__).parents[1] / "colab" / "minimind_colab.py"
SPEC = importlib.util.spec_from_file_location("minimind_colab", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class CommandSetupTests(unittest.TestCase):
    def test_preflight_bf16_guidance_accepts_l4_and_a100(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")

        self.assertIn("such as an L4 or A100", source)

    @mock.patch.object(RUNNER.shutil, "which", return_value="/usr/local/bin/pip")
    def test_repairs_partial_venv_by_using_external_pip(self, which):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").touch()
            venv = root / ".colab-venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").touch()
            venv = venv.resolve()
            args = argparse.Namespace(venv=str(venv))

            with mock.patch.object(RUNNER.subprocess, "run") as run:
                RUNNER.command_setup(args, root)

            self.assertEqual(
                run.call_args_list,
                [
                    mock.call(
                        [
                            "/usr/local/bin/pip",
                            "--python",
                            str(venv / "bin" / "python"),
                            "install",
                            "--upgrade",
                            "pip",
                            "-r",
                            str(root / "requirements.txt"),
                        ],
                        check=True,
                    ),
                ],
            )
            which.assert_called_once_with("pip")


class NotebookSetupCellTests(unittest.TestCase):
    def test_repository_setup_cell_is_cli_only_and_repeatable(self):
        notebook_path = RUNNER_PATH.with_name("minimind_colab_learning.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        config_source = "".join(notebook["cells"][1]["source"])
        setup_source = "".join(notebook["cells"][2]["source"])

        self.assertIn("COLAB_PYTHON =", config_source)
        self.assertIn("COLAB_RUNNER =", config_source)
        self.assertIn("REQUIRE_A100 = False", config_source)
        self.assertIn("ZERO_BATCH_SIZE = 8", config_source)
        self.assertIn("os.environ.update", config_source)
        self.assertTrue(setup_source.startswith("%%bash\nset -euo pipefail\n"))
        self.assertIn('if [ -d "$ROOT/.git" ]; then', setup_source)
        self.assertIn("git -C", setup_source)
        self.assertIn("git clone", setup_source)
        self.assertIn('python "$COLAB_RUNNER" --root "$ROOT" setup', setup_source)
        self.assertNotIn("from pathlib", setup_source)
        self.assertNotIn("subprocess", setup_source)
        self.assertNotIn("COLAB_PYTHON =", setup_source)
        self.assertNotIn("%cd", setup_source)
        self.assertLessEqual(max(map(len, setup_source.splitlines())), 100)

    def test_l4_profile_does_not_require_a100_and_uses_safe_zero_batch(self):
        notebook_path = RUNNER_PATH.with_name("minimind_colab_learning.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

        self.assertIn("preflight {PREFLIGHT_ACCELERATOR_ARG}", source)
        self.assertNotIn("preflight --require-a100", source)
        self.assertEqual(source.count("--batch-size {ZERO_BATCH_SIZE}"), 2)

    def test_non_git_root_stops_before_setup(self):
        notebook_path = RUNNER_PATH.with_name("minimind_colab_learning.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        setup_command = "".join(notebook["cells"][2]["source"]).removeprefix("%%bash\n")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sentinel = root.parent / f"{root.name}-setup-ran"
            probe = setup_command.rsplit("\npython ", 1)[0] + f"\ntouch '{sentinel}'\n"
            environment = os.environ | {
                "ROOT": str(root),
                "REPOSITORY_REF": "test-ref",
                "REPOSITORY_URL": "test-url",
                "COLAB_RUNNER": str(root / "colab" / "minimind_colab.py"),
            }
            result = subprocess.run(
                ["/bin/zsh", "-c", probe],
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
