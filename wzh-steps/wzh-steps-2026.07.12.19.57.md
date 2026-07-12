# MiniMind Colab CLI-only setup cell - command evidence

| Field | Value |
|---|---|
| Round | 4 |
| Status | Complete; reviewed and committed |
| Working directory | `/Users/zhengwan/Desktop/dev/minimind` |
| Shell | `zsh` locally; Colab commands use IPython CLI magics |
| Started | 2026-07-12 19:57:08 +0800 |

## Requirement

The repository/setup notebook cell must not mix Python control flow with CLI commands. The configuration cell owns values such as `ROOT`, `COLAB_PYTHON`, and `COLAB_RUNNER`; the execution cell must be one fail-fast CLI invocation containing Git, shell `cd`, and the runner CLI.

## Red regression

```zsh
python -m unittest discover -s tests -p 'test_minimind_colab.py' -v
```

The new CLI-only assertion failed against the mixed `Path`/`subprocess`/magic cell. Exit code 1; [complete stdout and stderr](files/wzh-steps-2026-07-12-19-57/regression-red.txt).

The notebook containing the supplied successful dependency-install output is preserved as [minimind_colab_learning.supplied-output.ipynb](../wzh-solution/files/wzh-solution-2026-07-12-19-57/minimind_colab_learning.supplied-output.ipynb). The two-test regression source that drove the first red/green cycle is preserved as [test_minimind_colab.regression-source.py](../wzh-solution/files/wzh-solution-2026-07-12-19-57/test_minimind_colab.regression-source.py); it is not described as a distinct failing implementation artifact because the same test source becomes green when production changes.

## Implementation

The configuration cell now defines:

```python
COLAB_PYTHON = f'{ROOT}/.colab-venv/bin/python'
```

The execution cell is CLI-only:

```text
!set -e; if test -d "{ROOT}/.git"; then git -C "{ROOT}" fetch --depth 1 origin "{REPOSITORY_REF}" && git -C "{ROOT}" checkout --detach FETCH_HEAD; elif test -e "{ROOT}"; then printf '%s\n' "ERROR: {ROOT} exists but is not a Git checkout" >&2; exit 1; else git clone --depth 1 --branch "{REPOSITORY_REF}" "{REPOSITORY_URL}" "{ROOT}"; fi; cd "{ROOT}"; python "{COLAB_RUNNER}" --root "{ROOT}" setup
```

Review found that separate IPython `!`, `%cd`, and setup lines did not short-circuit after a failed shell command. The final cell uses one `set -e` CLI invocation, and all later runner commands use absolute runner/root arguments. A real zsh regression now verifies that an existing non-Git root exits nonzero without executing a sentinel setup action.

The previous output was archived, then stale outputs and execution counts were cleared with:

```zsh
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace colab/minimind_colab_learning.ipynb
```

The command exited 0. It emitted a non-fatal `MissingIDFieldWarning` while normalizing old notebook cells and wrote the cleaned notebook successfully. [Complete stdout, stderr, and exit code](files/wzh-steps-2026-07-12-19-57/notebook-clear-output.txt).

## Validation

### Full focused test suite

```zsh
python -m unittest discover -s tests -p 'test_minimind_colab.py' -v
```

Exit code 0; [complete stdout and stderr](files/wzh-steps-2026-07-12-19-57/regression-green.txt). Three tests pass: pip-less venv repair, CLI-only/repeatable structure, and non-Git-root fail-fast behavior.

### Notebook structure

The validation parsed all Python portions after replacing IPython command/magic lines with `pass`, required all outputs/execution counts to be clear, and asserted that the setup cell is a single `set -e` CLI command containing setup with no `%cd` fallthrough.

```zsh
python -c "import ast,json,pathlib; n=json.loads(pathlib.Path('colab/minimind_colab_learning.ipynb').read_text()); code=[c for c in n['cells'] if c['cell_type']=='code']; transformed=[''.join((' '*(len(line)-len(line.lstrip()))+'pass\\n') if line.lstrip().startswith(('!','%')) else line for line in c['source']) for c in code]; [ast.parse(src, filename=f'<cell-{i}>') for i,src in enumerate(transformed)]; assert all(c.get('execution_count') is None and not c.get('outputs') for c in code); setup=n['cells'][2]['source']; assert len(setup)==1 and setup[0].startswith('!set -e;') and '; python ' in setup[0] and '%cd' not in setup[0]; print(f'notebook_schema=ok code_cells={len(code)} syntax=ok outputs_clean=ok setup_cli_only=ok fail_fast=ok')"
```

Exit code 0; [complete output](files/wzh-steps-2026-07-12-19-57/notebook-validation.txt).

### Python compilation

```zsh
python -m py_compile colab/minimind_colab.py tests/test_minimind_colab.py
```

Exit code 0; [complete empty output plus exit code](files/wzh-steps-2026-07-12-19-57/pycompile.txt).

## Exact artifacts

```zsh
cp colab/minimind_colab_learning.ipynb wzh-solution/files/wzh-solution-2026-07-12-19-57/minimind_colab_learning.ipynb
cp colab/README.md wzh-solution/files/wzh-solution-2026-07-12-19-57/README.md
cp tests/test_minimind_colab.py wzh-solution/files/wzh-solution-2026-07-12-19-57/test_minimind_colab.py
shasum -a 256 colab/minimind_colab_learning.ipynb colab/README.md tests/test_minimind_colab.py
wc -c colab/minimind_colab_learning.ipynb colab/README.md tests/test_minimind_colab.py
```

All commands exited 0. See [SHA-256 values](files/wzh-steps-2026-07-12-19-57/artifact-sha256.txt) and [byte counts](files/wzh-steps-2026-07-12-19-57/artifact-bytes.txt).

## Command metadata index

All entries used cwd `/Users/zhengwan/Desktop/dev/minimind`, local `zsh`, and timestamp timezone `+0800`.

| Timestamp | Command | Exit | Complete stdout/stderr |
|---|---|---:|---|
| 2026-07-12 19:57:08 | Focused regression before implementation | 1 | [regression-red.txt](files/wzh-steps-2026-07-12-19-57/regression-red.txt) |
| 2026-07-12 19:59 | `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace ...` | 0 | [notebook-clear-output.txt](files/wzh-steps-2026-07-12-19-57/notebook-clear-output.txt) |
| 2026-07-12 20:04 | Final focused regression | 0 | [regression-green.txt](files/wzh-steps-2026-07-12-19-57/regression-green.txt) |
| 2026-07-12 20:04 | Final notebook structure validation | 0 | [notebook-validation.txt](files/wzh-steps-2026-07-12-19-57/notebook-validation.txt) |
| 2026-07-12 20:04 | `py_compile` | 0 | [pycompile.txt](files/wzh-steps-2026-07-12-19-57/pycompile.txt); stdout/stderr empty before recorded exit code. |
| 2026-07-12 20:05:24 | Final `cp`, `shasum`, and `wc` artifact capture | 0 each | [hashes](files/wzh-steps-2026-07-12-19-57/artifact-sha256.txt), [byte counts](files/wzh-steps-2026-07-12-19-57/artifact-bytes.txt); `cp` stdout/stderr empty. |
| 2026-07-12 20:07 | Full `test_*.py` discovery after review fixes | 0 | [full-test-final.txt](files/wzh-steps-2026-07-12-19-57/full-test-final.txt) |
| 2026-07-12 20:07 | Final diff whitespace check excluding raw evidence | 0 | [diff-check-final.txt](files/wzh-steps-2026-07-12-19-57/diff-check-final.txt); stdout/stderr empty before recorded exit code. |

The final review commands were run exactly as follows:

```zsh
python -m unittest discover -s tests -p 'test_*.py' -v
```

```zsh
git diff --check -- . ':(exclude)wzh-steps/files/**'
```
