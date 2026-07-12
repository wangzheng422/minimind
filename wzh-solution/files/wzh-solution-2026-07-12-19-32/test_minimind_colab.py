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
    def test_repairs_partial_venv_without_ensurepip_and_uses_visible_pip(self):
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
                            RUNNER.sys.executable,
                            "-m",
                            "venv",
                            "--without-pip",
                            "--system-site-packages",
                            str(venv),
                        ],
                        check=True,
                    ),
                    mock.call(
                        [
                            str(venv / "bin" / "python"),
                            "-m",
                            "pip",
                            "install",
                            "--upgrade",
                            "-r",
                            str(root / "requirements.txt"),
                        ],
                        check=True,
                    ),
                ],
            )


if __name__ == "__main__":
    unittest.main()
