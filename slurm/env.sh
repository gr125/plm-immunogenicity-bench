# Shared environment setup. Sourced by every job script; safe to source by hand.
#
#   source slurm/env.sh
#
# Establishes, in order: the repository root, where the embedding pickles live,
# the project virtualenv, and PYTHONPATH. Relies on no conda installation and no
# pre-existing named environment -- run slurm/bootstrap.sh once to create the
# venv, then every job uses it.
#
# Override any of these before sourcing:
#   PLMBENCH_ROOT     repository root          (default: this file's parent dir)
#   PLMBENCH_PICKLES  embedding pickle tree    (default: same as root)
#   PLMBENCH_VENV     virtualenv to activate   (default: $PLMBENCH_ROOT/.venv)

# Resolve the repo root from this file's own location, so the scripts work from
# any clone without editing a path.
if [ -z "${PLMBENCH_ROOT:-}" ]; then
    _env_sh="${BASH_SOURCE[0]:-$0}"
    PLMBENCH_ROOT="$(cd "$(dirname "$_env_sh")/.." && pwd)"
fi
export PLMBENCH_ROOT
export PLMBENCH_PICKLES="${PLMBENCH_PICKLES:-$PLMBENCH_ROOT}"
export PLMBENCH_VENV="${PLMBENCH_VENV:-$PLMBENCH_ROOT/.venv}"

if [ ! -f "$PLMBENCH_ROOT/configs/paths.yaml" ]; then
    echo "ERROR: $PLMBENCH_ROOT does not look like the repository" >&2
    echo "       (no configs/paths.yaml under it)." >&2
    echo "       Submit from the repo root, or set PLMBENCH_ROOT explicitly." >&2
    exit 1
fi

cd "$PLMBENCH_ROOT"
mkdir -p logs

if [ ! -x "$PLMBENCH_VENV/bin/python" ]; then
    echo "ERROR: no virtualenv at $PLMBENCH_VENV" >&2
    echo "" >&2
    echo "Create it once on a login node (it needs network access):" >&2
    echo "    bash $PLMBENCH_ROOT/slurm/bootstrap.sh          # analysis only" >&2
    echo "    bash $PLMBENCH_ROOT/slurm/bootstrap.sh --embed  # + torch/transformers/esm" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$PLMBENCH_VENV/bin/activate"

# Run from src/ as well, so the package works whether or not it installed.
export PYTHONPATH="$PLMBENCH_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "root    : $PLMBENCH_ROOT"
echo "pickles : $PLMBENCH_PICKLES"
echo "venv    : $PLMBENCH_VENV"
echo "python  : $(python -V 2>&1)"
