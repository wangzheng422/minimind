import argparse
import importlib.util
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
    def test_repository_clone_is_guarded_for_repeated_cell_execution(self):
        import json

        notebook_path = RUNNER_PATH.with_name("minimind_colab_learning.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        setup_source = "".join(notebook["cells"][2]["source"])

        self.assertIn("if not root.exists():", setup_source)
        self.assertIn("subprocess.run", setup_source)
        self.assertNotIn("!git clone", setup_source)


if __name__ == "__main__":
    unittest.main()
