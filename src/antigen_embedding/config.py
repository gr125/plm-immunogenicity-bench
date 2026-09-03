"""Configuration loading and path resolution.

Every script in this package takes its paths and parameters from configs/,
never from a hardcoded prefix. Loading is:

    from antigen_embedding.config import load_config
    cfg = load_config(variant="no_anchor")
    cfg.path("data.epitopes")          -> /abs/path/data/epitopes.csv.gz
    cfg.variant["slice"]["trim_start"] -> 2

The repository root is resolved once, in this order:

    1. $ANTIGEN_EMBEDDING_ROOT
    2. an explicit root= argument
    3. an absolute `root:` in configs/paths.yaml
    4. the parent of the configs/ directory

so the same configs run unchanged on a laptop and on the cluster.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# .../src/antigen_embedding/config.py -> repo root is three levels up
_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent.parent
DEFAULT_CONFIG_DIR = _REPO_ROOT / "configs"

ENV_ROOT = "ANTIGEN_EMBEDDING_ROOT"


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`.

    Mappings merge key by key; every other type (scalars, lists) is replaced
    wholesale, so a variant listing `embeddings:` replaces the base list rather
    than appending to it.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _dig(mapping: dict, dotted: str) -> Any:
    """Look up 'a.b.c' in nested dicts, raising a useful error if absent."""
    node: Any = mapping
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"no config key {dotted!r} (failed at {part!r})")
        node = node[part]
    return node


def _coerce(text: str) -> Any:
    """Parse a CLI override value using YAML rules, so 'null'/'2'/'true' work."""
    return yaml.safe_load(text)


@dataclass
class Config:
    """Resolved configuration for one run."""

    root: Path
    config_dir: Path
    paths: dict
    embeddings: dict
    variant: dict = field(default_factory=dict)

    # -- paths -------------------------------------------------------------

    def resolve(self, value: str | os.PathLike) -> Path:
        """Make one path absolute against the repository root."""
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    def path(self, dotted: str) -> Path:
        """Resolve a dotted key from paths.yaml, e.g. 'data.epitopes'."""
        return self.resolve(_dig(self.paths, dotted))

    def out_dir(self, dotted: str) -> Path:
        """Resolve a dotted key from the variant's `output:` block and mkdir it."""
        d = self.resolve(_dig(self.variant, dotted))
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- embeddings --------------------------------------------------------

    def embedding_spec(self, name: str) -> dict:
        try:
            spec = self.embeddings["embeddings"][name]
        except KeyError:
            known = ", ".join(sorted(self.embeddings["embeddings"]))
            raise KeyError(f"unknown embedding {name!r}; known: {known}") from None
        spec = dict(spec, name=name)
        spec["pickle"] = self.resolve(spec["pickle"])
        return spec

    def selected_embeddings(self) -> list[str]:
        """The variant's embeddings, in the registry's display order."""
        wanted = self.variant.get("embeddings") or list(self.embeddings["embeddings"])
        order = self.embeddings.get("order", [])
        ranked = sorted(wanted, key=lambda n: order.index(n) if n in order else len(order))
        return ranked

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a dotted key from the variant, falling back to `default`."""
        try:
            return _dig(self.variant, dotted)
        except KeyError:
            return default


def load_config(
    variant: str | os.PathLike | None = "full",
    root: str | os.PathLike | None = None,
    config_dir: str | os.PathLike | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load paths.yaml, embeddings.yaml and one variant into a Config.

    variant   name under configs/variants/ ("no_anchor") or a path to a YAML
              file. None loads paths and the registry only.
    root      overrides the repository root (highest precedence after the
              environment variable).
    overrides dotted variant keys to override, e.g. {"slice.trim_start": 2}.
    """
    cdir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    cdir = cdir.resolve()

    paths = _read_yaml(cdir / "paths.yaml")
    registry = _read_yaml(cdir / "embeddings.yaml")

    resolved_root = _resolve_root(root, paths, cdir)

    variant_cfg = _load_variant(variant, cdir) if variant is not None else {}
    for dotted, value in (overrides or {}).items():
        _set_dotted(variant_cfg, dotted, value)

    return Config(
        root=resolved_root,
        config_dir=cdir,
        paths=paths,
        embeddings=registry,
        variant=variant_cfg,
    )


def _resolve_root(root, paths: dict, config_dir: Path) -> Path:
    env = os.environ.get(ENV_ROOT)
    if root:
        return Path(root).expanduser().resolve()
    if env:
        return Path(env).expanduser().resolve()
    declared = paths.get("root")
    if declared and Path(declared).is_absolute():
        return Path(declared).resolve()
    return config_dir.parent


def _load_variant(variant, config_dir: Path, _seen: set | None = None) -> dict:
    """Load a variant file, applying its `extends:` chain first."""
    _seen = _seen or set()
    p = Path(variant)
    if not p.suffix:
        p = p.with_suffix(".yaml")
    if not p.is_absolute() and not p.is_file():
        p = config_dir / "variants" / p
    p = p.resolve()

    if p in _seen:
        raise ValueError(f"circular `extends:` chain through {p}")
    _seen.add(p)

    cfg = _read_yaml(p)
    parent_name = cfg.pop("extends", None)
    if parent_name:
        parent = _load_variant(parent_name, config_dir, _seen)
        cfg = _deep_merge(parent, cfg)
    return cfg


def _set_dotted(mapping: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = mapping
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def add_config_args(parser) -> None:
    """Attach the standard --variant/--root/--set flags to an ArgumentParser."""
    parser.add_argument(
        "--variant", default="full",
        help="variant config name under configs/variants/, or a path to a YAML file",
    )
    parser.add_argument(
        "--root", default=None,
        help=f"repository root (overrides ${ENV_ROOT} and configs/paths.yaml)",
    )
    parser.add_argument(
        "--config-dir", default=None,
        help="directory holding paths.yaml, embeddings.yaml and variants/",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a variant key, e.g. --set slice.trim_start=2 "
             "(repeatable; VALUE is parsed as YAML)",
    )


def config_from_args(args) -> Config:
    """Build a Config from a parser that was given add_config_args()."""
    overrides = {}
    for item in getattr(args, "overrides", []) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        overrides[key.strip()] = _coerce(raw.strip())
    return load_config(
        variant=getattr(args, "variant", "full"),
        root=getattr(args, "root", None),
        config_dir=getattr(args, "config_dir", None),
        overrides=overrides,
    )
