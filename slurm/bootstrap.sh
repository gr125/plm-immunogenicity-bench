#!/bin/bash
# Build the project's virtualenv. Run once, on a login node (needs network).
#
#   bash slurm/bootstrap.sh              # analysis dependencies only
#   bash slurm/bootstrap.sh --embed      # + torch, transformers, esm
#
# No conda, and no pre-existing named environment. The only prerequisite is a
# Python >= 3.10 interpreter; point at a specific one with PLMBENCH_PYTHON, or
# let the script find the newest available.
#
#   PLMBENCH_PYTHON=/usr/bin/python3.11 bash slurm/bootstrap.sh
#   PLMBENCH_VENV=/scratch/$USER/plmbench-venv bash slurm/bootstrap.sh
#
# Offline cluster? Point pip at a local wheelhouse:
#   PIP_ARGS="--no-index --find-links=/path/to/wheels" bash slurm/bootstrap.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PLMBENCH_VENV:-$ROOT/.venv}"
WITH_EMBED=0
[ "${1:-}" = "--embed" ] && WITH_EMBED=1

# ---- find an interpreter -------------------------------------------------
find_python() {
    if [ -n "${PLMBENCH_PYTHON:-}" ]; then echo "$PLMBENCH_PYTHON"; return; fi
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$c" >/dev/null 2>&1 && \
           "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            command -v "$c"; return
        fi
    done
    return 1
}

PY="$(find_python || true)"
if [ -z "$PY" ]; then
    echo "ERROR: no Python >= 3.10 found." >&2
    echo "Load one first (e.g. 'module load python/3.11'), or set PLMBENCH_PYTHON." >&2
    exit 1
fi
echo "interpreter : $PY  ($("$PY" -V 2>&1))"
echo "venv        : $VENV"

# ---- create and populate -------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    "$PY" -m venv "$VENV"
else
    echo "(reusing the existing venv; delete it to rebuild from scratch)"
fi

# A fresh venv can ship a pip too old for pyproject-only projects. Upgrading it
# here is what makes `pip install -e .` work regardless of the system pip.
# shellcheck disable=SC2086
"$VENV/bin/python" -m pip install --upgrade ${PIP_ARGS:-} pip setuptools wheel

if [ "$WITH_EMBED" = "1" ]; then
    echo "installing analysis + embedding dependencies"
    echo "NOTE: torch is installed from PyPI, which may not match this cluster's"
    echo "      CUDA. If the GPU is not detected, reinstall torch from the index"
    echo "      for your CUDA version, e.g."
    echo "      $VENV/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121"
    # shellcheck disable=SC2086
    "$VENV/bin/pip" install ${PIP_ARGS:-} -e "$ROOT[embed]"
else
    # shellcheck disable=SC2086
    "$VENV/bin/pip" install ${PIP_ARGS:-} -e "$ROOT"
fi

# ---- verify --------------------------------------------------------------
echo
echo "verifying imports"
"$VENV/bin/python" - <<'PYEOF'
import importlib, sys
required = ["pandas", "numpy", "yaml", "sklearn", "umap", "seaborn", "matplotlib", "skbio"]
missing = []
for m in required:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e}")
for m in ["plmbench.config", "plmbench.analysis.umap_runner", "plmbench.check"]:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e}")
if missing:
    print("  FAILED:"); [print("   ", x) for x in missing]; sys.exit(1)
print(f"  {len(required)} runtime deps + plmbench import cleanly")
PYEOF

echo
echo "Done. Every job script now finds this venv automatically."
echo "Next:  PLMBENCH_PICKLES=/mnt/bioadhoc/Groups/Peters/Self-similarity \\"
echo "         $VENV/bin/python -m plmbench.check --deep"
