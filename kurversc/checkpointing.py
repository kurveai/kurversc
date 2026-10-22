"""Durable, contract-scoped checkpoints for expensive search trials."""

from __future__ import annotations

import copy
from dataclasses import asdict
from functools import lru_cache
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import pickle
import platform
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from .search import GraphConfig, Trial


CHECKPOINT_FORMAT_VERSION = 1
SEARCH_CONTRACT_VERSION = "kurversc-search-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _package_source_fingerprint(name: str) -> str:
    """Fingerprint editable source trees as well as installed packages."""

    module = import_module(name)
    module_path = Path(module.__file__ or "").resolve()
    root = module_path.parent if module_path.name == "__init__.py" else module_path
    paths = sorted(root.rglob("*.py")) if root.is_dir() else [root]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root) if root.is_dir() else Path(path.name)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def search_runtime_identity() -> dict[str, Any]:
    """Return the execution identity that makes cached trial scores valid.

    Distribution versions alone are insufficient during editable development:
    two source trees can both report the same package version while generating
    different feature frames.  Source fingerprints prevent those trials from
    being silently mixed.
    """

    return {
        "python": platform.python_version(),
        "packages": {
            name: _distribution_version(name)
            for name in (
                "catboost",
                "duckdb",
                "numpy",
                "pandas",
                "scikit-learn",
            )
        },
        "kurversc_source": _package_source_fingerprint("kurversc"),
        "graphreduce_source": _package_source_fingerprint("graphreduce"),
        "graphreduce_version": getattr(
            import_module("graphreduce"), "__version__", None
        ),
    }


class SearchCheckpoint:
    """Store one atomic file per completed trial under a search contract.

    The contract fingerprint isolates incompatible datasets and search settings.
    Individual trial files let forked workers persist results independently, so
    completed work survives termination of the parent process or another worker.
    Checkpoints use pickle because a frozen GraphReduce execution plan can contain
    typed values that cannot be losslessly represented as JSON. Callers must only
    resume from checkpoint directories they trust.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        contract: Mapping[str, Any],
    ) -> None:
        self.root = Path(root).expanduser()
        self.contract = {
            "search_contract_version": SEARCH_CONTRACT_VERSION,
            "runtime": search_runtime_identity(),
            **dict(contract),
        }
        self.contract_fingerprint = _digest(self.contract)
        self.run_directory = self.root / self.contract_fingerprint
        self.trial_directory = self.run_directory / "trials"
        self.trial_directory.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _write_manifest(self) -> None:
        path = self.run_directory / "manifest.json"
        if path.exists():
            return
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "contract_fingerprint": self.contract_fingerprint,
            "contract": self.contract,
        }
        self._atomic_bytes(path, (_canonical_json(payload) + "\n").encode("utf-8"))

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def trial_key(
        config: GraphConfig,
        *,
        stage: str,
        sample_rows: int | None,
    ) -> str:
        return _digest(
            {
                "config": asdict(config),
                "stage": stage,
                "sample_rows": sample_rows,
            }
        )

    def _trial_path(
        self,
        config: GraphConfig,
        *,
        stage: str,
        sample_rows: int | None,
    ) -> Path:
        key = self.trial_key(config, stage=stage, sample_rows=sample_rows)
        return self.trial_directory / f"{key}.pkl"

    def load(
        self,
        config: GraphConfig,
        *,
        stage: str,
        sample_rows: int | None,
    ) -> Trial | None:
        path = self._trial_path(
            config,
            stage=stage,
            sample_rows=sample_rows,
        )
        if not path.exists():
            return None
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - documented trusted cache
        if not isinstance(payload, dict):
            return None
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            return None
        if payload.get("contract_fingerprint") != self.contract_fingerprint:
            return None
        trial = payload.get("trial")
        if not isinstance(trial, Trial):
            return None
        if trial.config != config or trial.stage != stage:
            return None
        if trial.sample_rows != sample_rows or trial.status != "completed":
            return None
        trial.model = None
        return trial

    def save(self, trial: Trial) -> Path | None:
        """Atomically persist a successful trial, returning its path."""

        if trial.status != "completed":
            return None
        stored = copy.copy(trial)
        stored.model = None
        payload = pickle.dumps(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "contract_fingerprint": self.contract_fingerprint,
                "trial": stored,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        path = self._trial_path(
            stored.config,
            stage=stored.stage,
            sample_rows=stored.sample_rows,
        )
        self._atomic_bytes(path, payload)
        return path
