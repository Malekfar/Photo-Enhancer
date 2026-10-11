"""Download and verify weights listed in weights.yaml.

    python -m scripts.download_weights                 # all entries
    python -m scripts.download_weights retinexformer_fivek zerodce_pp
    python -m scripts.download_weights --verify-only   # checksum existing files, no network
    python -m scripts.download_weights --prefetch-iqa  # also cache pyiqa NIQE/BRISQUE files

Exit code is non-zero if any requested pinned weight fails to download or verify.
Unpinned entries (empty sha256) are skipped with a warning unless
LLE_ALLOW_UNPINNED_WEIGHTS=true (or --allow-unpinned), in which case the hash is pinned on
first download into <weights_dir>/pins.json and printed so it can be copied to weights.yaml.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.errors import WeightsError
from app.logging_setup import setup_logging
from app.weights import download, expected_sha256, load_manifest, verify_file

log = logging.getLogger("download_weights")


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="manifest entries (default: all)")
    ap.add_argument("--weights-dir", type=Path, default=s.weights_dir)
    ap.add_argument("--manifest", type=Path, default=s.weights_manifest)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--allow-unpinned", action="store_true", default=s.allow_unpinned_weights)
    ap.add_argument("--prefetch-iqa", action="store_true", help="instantiate pyiqa NIQE/BRISQUE to cache their files")
    args = ap.parse_args(argv)
    setup_logging(s.log_level, s.log_json)

    manifest = load_manifest(args.manifest)
    names = args.names or list(manifest)
    unknown = [n for n in names if n not in manifest]
    if unknown:
        log.error("unknown weight names", extra={"names": unknown, "available": list(manifest)})
        return 2

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for name in names:
        spec = manifest[name]
        expected, _ = expected_sha256(spec, args.weights_dir)
        if expected is None and not args.allow_unpinned:
            log.warning(
                "skipping unpinned weights (set LLE_ALLOW_UNPINNED_WEIGHTS=true to download and pin)",
                extra={"weights": name},
            )
            continue
        try:
            if args.verify_only:
                path = verify_file(spec, args.weights_dir, allow_unpinned=args.allow_unpinned)
            else:
                path = download(spec, args.weights_dir, allow_unpinned=args.allow_unpinned)
            digest, origin = expected_sha256(spec, args.weights_dir)
            log.info("weights OK", extra={"weights": name, "path": str(path), "sha256": digest, "pinned_by": origin})
        except WeightsError as exc:
            log.error("weights FAILED", extra={"weights": name, "error": str(exc)})
            failed.append(name)

    if args.prefetch_iqa:
        try:
            import pyiqa  # noqa: PLC0415

            for metric in ("niqe", "brisque"):
                pyiqa.create_metric(metric, device="cpu", as_loss=False)
            log.info("pyiqa NIQE/BRISQUE files cached")
        except Exception as exc:  # noqa: BLE001 - IQA is optional for serving
            log.warning("could not prefetch pyiqa metrics", extra={"error": f"{type(exc).__name__}: {exc}"})

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
