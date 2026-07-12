# MiniMind Colab repeatable setup repair - command evidence

| Field | Value |
|---|---|
| Round | 3 |
| Status | Complete |
| Working directory | `/Users/zhengwan/Desktop/dev/minimind` |
| Shell | `zsh` |
| Started | 2026-07-12 19:39:31 +0800 |
| Environment | Local conda Python 3.12; disposable pip-less venvs; no live Colab GPU runtime |

## Correction to round 2

Round 2 concluded that `--system-site-packages` would make host pip importable through `.colab-venv/bin/python -m pip`. The supplied Colab output disproved that assumption: the venv was created, but its interpreter reported `No module named pip`. This round preserves the earlier record and corrects the implementation by invoking host `pip --python <venv-python>` instead.

The supplied notebook also showed an independent failure: unconditional `git clone` rejected the existing non-empty `/content/minimind` directory. The notebook setup cell is now repeatable and updates an existing checkout.

## Red-capable regression loop

Command:

```zsh
python -m unittest discover -s tests -p 'test_minimind_colab.py' -v
```

Before the fix, both focused tests failed: the runner recreated the venv and invoked its missing pip module, while the notebook still contained unconditional `!git clone`. Exit code 1; [complete stdout and stderr](files/wzh-steps-2026-07-12-19-39/regression-red.txt).

The exact supplied failure notebook is preserved as [minimind_colab_learning.supplied-failure.ipynb](../wzh-solution/files/wzh-solution-2026-07-12-19-39/minimind_colab_learning.supplied-failure.ipynb). The pre-fix runner and red test are preserved in the same artifact directory.

## Hypothesis probes and fix validation

### Focused regression

```zsh
python -m unittest discover -s tests -p 'test_minimind_colab.py' -v
```

Exit code 0; [complete stdout and stderr](files/wzh-steps-2026-07-12-19-39/regression-green.txt).

### External pip targeting a pip-less venv

```zsh
tmpdir=$(mktemp -d /private/tmp/minimind-pipless.XXXXXX)
python -m venv --without-pip --system-site-packages "$tmpdir"
pip --python "$tmpdir/bin/python" --version
```

Exit code 0. This proves the host pip supports the same target-interpreter interface used by `command_setup`, without requiring `python -m pip` inside the target venv. [Complete output](files/wzh-steps-2026-07-12-19-39/external-pip-probe.txt).

### Python compilation

```zsh
python -m py_compile colab/minimind_colab.py tests/test_minimind_colab.py
```

Exit code 0; [complete empty stdout and stderr plus exit code](files/wzh-steps-2026-07-12-19-39/pycompile.txt).

### Notebook validation

```zsh
python -c "import ast,json,pathlib; p=pathlib.Path('colab/minimind_colab_learning.ipynb'); n=json.loads(p.read_text()); code=[c for c in n['cells'] if c['cell_type']=='code']; transformed=[]; [transformed.append(''.join((' '*(len(line)-len(line.lstrip()))+'pass\\n') if line.lstrip().startswith(('!','%')) else line for line in c['source'])) for c in code]; [ast.parse(src, filename=f'<cell-{i}>') for i,src in enumerate(transformed)]; assert all(c.get('execution_count') is None and not c.get('outputs') for c in code); setup=''.join(n['cells'][2]['source']); assert 'if not root.exists():' in setup and '!git clone' not in setup and 'fetch' in setup; print(f'notebook_schema=ok code_cells={len(code)} syntax=ok outputs_clean=ok repeated_setup=ok')"
```

Exit code 0; [complete output](files/wzh-steps-2026-07-12-19-39/notebook-validation.txt).

## Exact final artifacts

The following commands archived the final sources:

```zsh
cp colab/minimind_colab.py wzh-solution/files/wzh-solution-2026-07-12-19-39/minimind_colab.py
cp colab/minimind_colab_learning.ipynb wzh-solution/files/wzh-solution-2026-07-12-19-39/minimind_colab_learning.ipynb
cp colab/README.md wzh-solution/files/wzh-solution-2026-07-12-19-39/README.md
cp tests/test_minimind_colab.py wzh-solution/files/wzh-solution-2026-07-12-19-39/test_minimind_colab.py
shasum -a 256 colab/minimind_colab.py colab/minimind_colab_learning.ipynb colab/README.md tests/test_minimind_colab.py
wc -c colab/minimind_colab.py colab/minimind_colab_learning.ipynb colab/README.md tests/test_minimind_colab.py
```

All commands exited 0. See [SHA-256 values](files/wzh-steps-2026-07-12-19-39/artifact-sha256.txt) and [byte counts](files/wzh-steps-2026-07-12-19-39/artifact-bytes.txt).

## Final scope checks

```zsh
git diff --check -- colab/minimind_colab.py colab/minimind_colab_learning.ipynb colab/README.md tests/test_minimind_colab.py wzh-todo.md wzh-steps/wzh-steps-2026.07.12.19.39.md wzh-solution/wzh-solution-2026.07.12.19.39.md
git status --short
git diff --stat
rg -n '\[DEBUG-' colab tests
```

The whitespace check exited 0; [complete output](files/wzh-steps-2026-07-12-19-39/diff-check.txt). The final scope and debug-instrumentation audit is preserved in [final-scope.txt](files/wzh-steps-2026-07-12-19-39/final-scope.txt). No `[DEBUG-...]` instrumentation remained. Round 2 files and the user's personal notebook configuration remain preserved in the worktree.
