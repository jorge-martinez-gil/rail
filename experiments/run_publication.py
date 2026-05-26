"""Canonical runner for RAIL publication artifacts.

This script intentionally keeps the existing experiment modules intact while
providing one stable entrypoint for reviewers and future paper revisions.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_FIGURE_FORMATS = ("png", "pdf", "svg")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_artifact_index(root: Path, metadata: dict[str, Any]) -> str:
    try:
        from .publication_artifacts import write_artifact_index
    except ImportError:
        from publication_artifacts import write_artifact_index

    return write_artifact_index(root, metadata=metadata)


def _import_real_data_module():
    try:
        from . import experiments_ae
    except ImportError:
        import experiments_ae

    return experiments_ae


def _import_self_contained_module():
    try:
        from . import rail_paper
    except ImportError:
        import rail_paper

    return rail_paper


def run_real_data(output_dir: Path, cache_dir: Path) -> None:
    """Run the real-data experiment pipeline."""

    module = _import_real_data_module()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    module.OUTDIR = output_dir
    module.CACHEDIR = cache_dir
    module.main()


def run_self_contained(output_dir: Path, runs: int, seed: int) -> dict[str, Any]:
    """Run the generated benchmark-like pipeline with stronger baselines."""

    module = _import_self_contained_module()
    output_dir.mkdir(parents=True, exist_ok=True)
    return module.run_benchmark(runs=runs, output_dir=str(output_dir), seed=seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RAIL publication artifact pipelines."
    )
    parser.add_argument(
        "--mode",
        choices=["real", "self-contained", "both"],
        default="self-contained",
        help="Which experiment pipeline to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="publication_outputs",
        help="Directory for generated tables, figures, and manifest files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="_cache",
        help="Directory for downloaded external datasets used by the real-data pipeline.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of self-contained benchmark repetitions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed for reproducible runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema": "RAIL.publication_run_manifest.v1",
        "started_at": _utc_now(),
        "completed_at": None,
        "status": "running",
        "mode": args.mode,
        "runs": args.runs,
        "seed": args.seed,
        "output_dir": str(root),
        "cache_dir": args.cache_dir,
        "python": sys.version,
        "platform": platform.platform(),
        "figure_formats": list(DEFAULT_FIGURE_FORMATS),
        "components": {},
    }

    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        if args.mode in {"real", "both"}:
            real_dir = root / "real_data"
            print(f"[RAIL] Starting real-data pipeline -> {real_dir}", flush=True)
            run_real_data(real_dir, Path(args.cache_dir))
            manifest["components"]["real_data"] = {
                "output_dir": str(real_dir),
                "status": "completed",
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[RAIL] Completed real-data pipeline -> {real_dir}", flush=True)

        if args.mode in {"self-contained", "both"}:
            self_dir = root / "self_contained"
            print(
                f"[RAIL] Starting self-contained stress tests "
                f"({args.runs} run(s)) -> {self_dir}",
                flush=True,
            )
            result = run_self_contained(self_dir, runs=args.runs, seed=args.seed)
            manifest["components"]["self_contained"] = {
                "output_dir": result["output_dir"],
                "summary_rows": len(result.get("summary", [])),
                "status": "completed",
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[RAIL] Completed self-contained stress tests -> {self_dir}", flush=True)

        manifest["status"] = "completed"
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["completed_at"] = _utc_now()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            manifest["artifact_index"] = _write_artifact_index(root, metadata=manifest)
        except Exception as index_exc:
            manifest["artifact_index_error"] = f"{type(index_exc).__name__}: {index_exc}"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
