# MiniMind Colab readable Bash setup cell - command evidence

| Field | Value |
|---|---|
| Round | 5 |
| Status | Complete; reviewed and committed |
| Working directory | `/Users/zhengwan/Desktop/dev/minimind` |
| Local shell | `zsh` |
| Target cell shell | Explicit `%%bash` |
| Started | 2026-07-12 20:36:13 +0800 |

## Requirement and preserved starting point

The prior fail-fast CLI implementation was functionally correct but compressed into one unreadable line. The exact one-line notebook is preserved as [minimind_colab_learning.one-line.ipynb](../wzh-solution/files/wzh-solution-2026-07-12-20-36/minimind_colab_learning.one-line.ipynb).

The updated regression source required a multi-line `%%bash` cell, `set -euo pipefail`, no Python control flow, and maximum line length of 100 characters. The initial command was:

```zsh
python -m unittest discover -s tests -p 'test_minimind_colab.py' -v
```

At 2026-07-12 20:36:13 +0800, cwd `/Users/zhengwan/Desktop/dev/minimind`, it exited 1. [Complete stdout and stderr](files/wzh-steps-2026-07-12-20-36/regression-red.txt).

## Final implementation

The configuration cell exports `REPOSITORY_URL`, `REPOSITORY_REF`, `ROOT`, and `COLAB_RUNNER` through `os.environ`. The separate execution cell is formatted as:

```bash
%%bash
set -euo pipefail

if [ -d "$ROOT/.git" ]; then
  git -C "$ROOT" fetch --depth 1 origin "$REPOSITORY_REF"
  git -C "$ROOT" checkout --detach FETCH_HEAD
elif [ -e "$ROOT" ]; then
  printf '%s\n' "ERROR: $ROOT exists but is not a Git checkout" >&2
  exit 1
else
  git clone \
    --depth 1 \
    --branch "$REPOSITORY_REF" \
    "$REPOSITORY_URL" \
    "$ROOT"
fi

cd "$ROOT"
python "$COLAB_RUNNER" --root "$ROOT" setup
```

## Validation

All commands below ran from `/Users/zhengwan/Desktop/dev/minimind` under local `zsh`.

### Focused/full local tests

```zsh
python -m unittest discover -s tests -p 'test_*.py' -v
```

Exit code 0. Three tests passed, including the real non-Git-root short-circuit probe. [Complete stdout and stderr](files/wzh-steps-2026-07-12-20-36/regression-green.txt).

### Notebook validation attempts

The first validation attempted IPython's transformer. Its complete, exact shell command is preserved in [notebook-validation-command.txt](files/wzh-steps-2026-07-12-20-36/notebook-validation-command.txt).

It exited 1 because the active project `python` does not have the `IPython` module. [Complete stdout and stderr](files/wzh-steps-2026-07-12-20-36/notebook-validation.txt). No project change resulted from this failed attempt.

The standard-library fallback parsed all non-Bash code cells, checked the Bash cell structure, cleared outputs, and maximum line length:

```zsh
python -c "import ast,json,pathlib; n=json.loads(pathlib.Path('colab/minimind_colab_learning.ipynb').read_text()); code=[c for c in n['cells'] if c['cell_type']=='code']; setup=''.join(n['cells'][2]['source']); assert setup.startswith('%%bash\\nset -euo pipefail\\n') and max(map(len,setup.splitlines())) <= 100; ordinary=[c for c in code if not ''.join(c['source']).startswith('%%bash\\n')]; transformed=[''.join((' '*(len(line)-len(line.lstrip()))+'pass\\n') if line.lstrip().startswith(('!','%')) else line for line in c['source']) for c in ordinary]; [ast.parse(src, filename=f'<cell-{i}>') for i,src in enumerate(transformed)]; assert all(c.get('execution_count') is None and not c.get('outputs') for c in code); print(f'notebook_schema=ok code_cells={len(code)} python_syntax=ok bash_structure=ok outputs_clean=ok multiline_bash=ok max_line={max(map(len,setup.splitlines()))}')"
```

Exit code 0; [complete output](files/wzh-steps-2026-07-12-20-36/notebook-validation-fallback.txt). The longest setup-cell line is 67 characters.

### Bash-native syntax validation

```zsh
python -c "import json,pathlib; n=json.loads(pathlib.Path('colab/minimind_colab_learning.ipynb').read_text()); print(''.join(n['cells'][2]['source'][1:]), end='')" | bash -n
```

Exit code 0; [complete empty output plus exit code](files/wzh-steps-2026-07-12-20-36/bash-syntax.txt).

### Python compilation

```zsh
python -m py_compile tests/test_minimind_colab.py
```

Exit code 0; [complete empty output plus exit code](files/wzh-steps-2026-07-12-20-36/pycompile.txt).

## Exact artifacts

At the end of implementation, the final notebook, README, and test were copied under `wzh-solution/files/wzh-solution-2026-07-12-20-36/` and measured with:

```zsh
shasum -a 256 colab/minimind_colab_learning.ipynb colab/README.md tests/test_minimind_colab.py
wc -c colab/minimind_colab_learning.ipynb colab/README.md tests/test_minimind_colab.py
```

All artifact commands exited 0. See [SHA-256 values](files/wzh-steps-2026-07-12-20-36/artifact-sha256.txt) and [byte counts](files/wzh-steps-2026-07-12-20-36/artifact-bytes.txt).

## Final review checks

```zsh
python -m unittest discover -s tests -p 'test_*.py' -v
git diff --check -- . ':(exclude)wzh-steps/files/**'
```

Both commands exited 0. See [final full-test output](files/wzh-steps-2026-07-12-20-36/full-test-final.txt) and [final diff-check output](files/wzh-steps-2026-07-12-20-36/diff-check-final.txt). Standards/spec review found no remaining issue after the complete failed-validation command was archived and linked.
