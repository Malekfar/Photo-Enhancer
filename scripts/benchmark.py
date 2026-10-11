"""Compare enhancement backends on quality and latency.

    python -m scripts.benchmark                          # synthetic set, all loaded backends
    python -m scripts.benchmark --device cuda --iqa      # + NIQE/BRISQUE of outputs
    python -m scripts.benchmark --images ./my_refs       # add your own well-exposed references
    python -m scripts.benchmark --lol-dir /data/LOL-v2/Real_captured
    python -m scripts.benchmark --json results.json

Quality: every well-exposed reference is synthetically darkened (gamma 2.2-3.5, exposure
scale, Poisson-Gaussian noise), run through the full service pipeline and compared to the
reference with PSNR / SSIM / LPIPS. Highlight safety: worst-case increase of the clipped
pixel share (output vs input). Latency: full pipeline wall time at the given megapixel sizes.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.evaluation import DEGRADATIONS, LPIPS, darken, load_lol_v2_pairs, make_product_scene, psnr, reference_images, ssim
from app.logging_setup import setup_logging
from app.pipeline import EnhanceParams, run_pipeline
from app.pipeline.highlight import clipped_fraction
from app.service import ModelRegistry


@dataclass(frozen=True)
class Config:
    label: str
    method: str
    checkpoint: str | None
    preserve_color: bool


def build_configs(reg: ModelRegistry, methods: list[str] | None) -> list[Config]:
    out: list[Config] = []
    for method, cks in reg.enhancers.items():
        if methods and method not in methods:
            continue
        for ck in cks:
            for pc in (True, False):
                if method == "classical" and not pc:
                    continue  # classical is already luminance-based
                name = f"{method}:{ck}" if ck else method
                out.append(Config(f"{name} pc={'on' if pc else 'off'}", method, ck, pc))
    return out


def params_for(cfg: Config, s: Settings, denoise: str) -> EnhanceParams:
    return EnhanceParams(
        method=cfg.method,  # type: ignore[arg-type]
        checkpoint=cfg.checkpoint,
        strength=1.0,
        denoise=denoise,  # type: ignore[arg-type]
        denoise_strength=s.denoise_strength,
        highlight_low=s.highlight_low,
        highlight_high=s.highlight_high,
        preserve_color=cfg.preserve_color,
        sharpen=False,
        output_format="png",
        quality=95,
        strip_exif=True,
        iqa=False,
    )


def fmt(v: float | None, nd: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def mean_or_none(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(statistics.fmean(xs)) if xs else None


def evaluate(reg, cfgs, s, pairs, lpips, iqa, denoise) -> dict:
    results = {}
    for cfg in cfgs:
        enh = reg.enhancer(cfg.method, cfg.checkpoint)
        p = params_for(cfg, s, denoise)
        rows = {"psnr": [], "ssim": [], "lpips": [], "niqe": [], "brisque": [], "clip_delta_pct": [], "ms": []}
        for _, low, ref in pairs:
            t0 = time.perf_counter()
            out = run_pipeline(low, p, reg.pipeline_config(), enh, reg.denoiser()).image
            rows["ms"].append((time.perf_counter() - t0) * 1000)
            rows["psnr"].append(psnr(out, ref))
            rows["ssim"].append(ssim(out, ref))
            rows["lpips"].append(lpips(out, ref) if lpips is not None else None)
            rows["clip_delta_pct"].append((clipped_fraction(out) - clipped_fraction(low)) * 100)
            if iqa:
                sc = reg.iqa.score(out)
                rows["niqe"].append(sc.get("niqe"))
                rows["brisque"].append(sc.get("brisque"))
        results[cfg.label] = {
            "psnr": mean_or_none(rows["psnr"]),
            "ssim": mean_or_none(rows["ssim"]),
            "lpips": mean_or_none(rows["lpips"]),
            "niqe": mean_or_none(rows["niqe"]),
            "brisque": mean_or_none(rows["brisque"]),
            "max_clip_delta_pct": max(rows["clip_delta_pct"]) if rows["clip_delta_pct"] else None,
            "mean_ms": mean_or_none(rows["ms"]),
        }
    return results


def baseline(pairs, lpips) -> dict:
    return {
        "psnr": mean_or_none([psnr(l, r) for _, l, r in pairs]),
        "ssim": mean_or_none([ssim(l, r) for _, l, r in pairs]),
        "lpips": mean_or_none([lpips(l, r) for _, l, r in pairs]) if lpips is not None else None,
    }


def latency(reg, cfgs, s, sizes_mp: list[float], repeats: int, denoise: str) -> dict:
    out: dict = {}
    for mp in sizes_mp:
        h = int(np.sqrt(mp * 1e6 * 3 / 4))
        w = int(h * 4 / 3)
        ref = cv2.resize(make_product_scene(), (w, h), interpolation=cv2.INTER_LINEAR)
        low = darken(ref, gamma=2.8, exposure=0.4, peak=400.0, read_noise=0.006, seed=0)
        for cfg in cfgs:
            enh = reg.enhancer(cfg.method, cfg.checkpoint)
            p = params_for(cfg, s, denoise)
            run_pipeline(low[:256, :256], p, reg.pipeline_config(), enh, reg.denoiser())  # warm-up
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                run_pipeline(low, p, reg.pipeline_config(), enh, reg.denoiser())
                times.append((time.perf_counter() - t0) * 1000)
            out.setdefault(cfg.label, {})[f"{mp:g}MP"] = statistics.median(times)
    return out


def print_quality(title: str, base: dict, res: dict, iqa: bool) -> None:
    print(f"\n### {title}\n")
    cols = ["config", "PSNR↑", "SSIM↑", "LPIPS↓"] + (["NIQE↓", "BRISQUE↓"] if iqa else []) + ["max Δclip %", "ms/img"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    print(f"| dark input (no processing) | {fmt(base['psnr'], 2)} | {fmt(base['ssim'])} | {fmt(base['lpips'])} |"
          + (" | |" if iqa else "") + " | |")
    for label, r in sorted(res.items(), key=lambda kv: -(kv[1]["psnr"] or 0)):
        row = [label, fmt(r["psnr"], 2), fmt(r["ssim"]), fmt(r["lpips"])]
        if iqa:
            row += [fmt(r["niqe"], 2), fmt(r["brisque"], 2)]
        row += [fmt(r["max_clip_delta_pct"], 3), fmt(r["mean_ms"], 0)]
        print("| " + " | ".join(row) + " |")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None, help="auto|cpu|cuda (default: LLE_DEVICE)")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--images", type=Path, default=None, help="extra well-exposed reference images")
    ap.add_argument("--lol-dir", type=Path, default=None, help="LOL-v2 Real_captured dir (Test/Low, Test/Normal)")
    ap.add_argument("--lol-limit", type=int, default=0, help="evaluate only the first N LOL pairs")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--latency-mp", type=float, nargs="*", default=[1.0, 12.0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--denoise", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--iqa", action="store_true", help="also compute NIQE/BRISQUE of outputs")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    overrides = {"iqa_enabled": args.iqa, "log_json": False, "log_level": "WARNING"}
    if args.device:
        overrides["device"] = args.device
    s = Settings(**overrides)
    setup_logging(s.log_level, s.log_json)
    reg = ModelRegistry(s)
    reg.load()
    cfgs = build_configs(reg, args.methods)
    if not cfgs:
        print("no backends loaded; failures:", reg.failures, file=sys.stderr)
        return 1
    print(f"device: {reg.device} | configs: {len(cfgs)} | load failures: {reg.failures or 'none'}")
    lpips = None if args.no_lpips else LPIPS(reg.device)
    if lpips is not None and not lpips.available:
        print(f"LPIPS unavailable: {lpips.error}")
        lpips = None

    report: dict = {"device": str(reg.device), "failures": reg.failures}
    refs = reference_images(args.images, args.max_side)
    pairs = [
        (f"{name}/{d.label}", darken(ref, gamma=d.gamma, exposure=d.exposure, peak=d.peak, read_noise=d.read_noise, seed=i), ref)
        for name, ref in refs.items()
        for i, d in enumerate(DEGRADATIONS)
    ]
    print(f"synthetic pairs: {len(pairs)} ({len(refs)} references x {len(DEGRADATIONS)} degradations)")
    base = baseline(pairs, lpips)
    res = evaluate(reg, cfgs, s, pairs, lpips, args.iqa, args.denoise)
    print_quality("Synthetic low-light (darkened references)", base, res, args.iqa)
    report["synthetic"] = {"baseline": base, "results": res}

    if args.lol_dir:
        lol = load_lol_v2_pairs(args.lol_dir)
        if args.lol_limit:
            lol = lol[: args.lol_limit]
        print(f"\nLOL-v2 pairs: {len(lol)}")
        lbase = baseline(lol, lpips)
        lres = evaluate(reg, cfgs, s, lol, lpips, args.iqa, args.denoise)
        print_quality("LOL-v2-real test", lbase, lres, args.iqa)
        report["lol_v2"] = {"baseline": lbase, "results": lres}

    if args.latency_mp:
        lat = latency(reg, cfgs, s, args.latency_mp, args.repeats, args.denoise)
        print(f"\n### Latency (median of {args.repeats}, full pipeline, device={reg.device}, denoise={args.denoise})\n")
        sizes = [f"{mp:g}MP" for mp in args.latency_mp]
        print("| config | " + " | ".join(f"{sz} (ms)" for sz in sizes) + " |")
        print("|" + "---|" * (len(sizes) + 1))
        for label, r in lat.items():
            print(f"| {label} | " + " | ".join(fmt(r[sz], 0) for sz in sizes) + " |")
        report["latency_ms"] = lat

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=lambda o: None))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
