"""
pipeline.py -- end-to-end runner: data generation → training → C++ build → benchmark

Usage:
    uv run python src/pipeline.py [options]

    --ort-root PATH     Path to ONNX Runtime tar.gz install (omit if using apt deb)
    --build-dir PATH    C++ build directory                 (default: cpp/build)
    --jobs N            Parallel build jobs                 (default: all cores)
    --skip-generate     Skip data generation if CSV already exists
    --skip-train        Skip training if model artifacts already exist
    --skip-build        Skip C++ binary build
    --skip-benchmark    Skip benchmark.py (just build/train)

Steps:
    1. src/generate.py   (via sys.executable -- same uv interpreter)
    2. src/train.py
    3. cmake configure + build  (inference + trt_inference if TRT found)
    4. src/benchmark.py

ORT_ROOT:
    Pass --ort-root only when ONNX Runtime was installed from the GitHub tar.gz.
    If you used the apt deb package (libonnxruntime-dev) omit this flag entirely;
    CMake will find the library via the system paths automatically.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Python interpreter running this script (the uv-managed venv)
PY = sys.executable

# src/pipeline.py lives one level below the repo root
REPO_ROOT = Path(__file__).parent.parent.resolve()
ONNX_DIR = REPO_ROOT / "artifacts" / "onnx"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(label: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {label}")
    print(f"{bar}")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command, streaming output live. Exits the pipeline on failure."""
    display = " ".join(str(c) for c in cmd)
    print(f"$ {display}\n")
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT)
    if result.returncode != 0:
        print(f"\n[pipeline] FAILED (exit {result.returncode}): {display}", file=sys.stderr)
        sys.exit(result.returncode)


def _elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}m"


def _resolve_ort_root(user_ort_root: str | None) -> str | None:
    """
    Resolve ORT_ROOT for CMake.

    Priority:
      1) explicit --ort-root argument
      2) ORT_ROOT environment variable
      3) $HOME/onnxruntime if it looks like a tar.gz install
    """
    if user_ort_root:
        return str(Path(os.path.expandvars(os.path.expanduser(user_ort_root))).resolve())

    env_ort_root = os.environ.get("ORT_ROOT")
    if env_ort_root:
        return str(Path(os.path.expandvars(os.path.expanduser(env_ort_root))).resolve())

    home_ort_root = Path.home() / "onnxruntime"
    has_header = (home_ort_root / "include" / "onnxruntime_cxx_api.h").exists()
    has_lib = (home_ort_root / "lib" / "libonnxruntime.so").exists()
    if has_header and has_lib:
        return str(home_ort_root.resolve())

    return None


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def step_generate(skip: bool) -> None:
    csv_path = REPO_ROOT / "data" / "input" / "data_signals.csv"
    if skip and csv_path.exists():
        print(f"[skip] data already exists at {csv_path}")
        return
    _step("Step 1 / 4 — Data generation")
    t0 = time.perf_counter()
    _run([PY, "src/generate.py"])
    print(f"\n[done] generate  ({_elapsed(t0)})")


def step_train(skip: bool) -> None:
    pth_files = [ONNX_DIR / f"{m}.pth" for m in ["NaiveLSTM", "ImprovedLSTM", "Transformer", "TCN"]]
    if skip and all(p.exists() for p in pth_files):
        print("[skip] model artifacts already exist in artifacts/onnx/")
        return
    _step("Step 2 / 4 — Model training + export")
    t0 = time.perf_counter()
    _run([PY, "src/train.py"])
    print(f"\n[done] train  ({_elapsed(t0)})")


def step_build(skip: bool, ort_root: str | None, build_dir: Path, jobs: int) -> None:
    inference_bin = build_dir / "inference"
    if skip and inference_bin.exists():
        print(f"[skip] C++ binary already exists at {inference_bin}")
        return
    _step("Step 3 / 4 — C++ build")
    build_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cmake_configure = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"]
    if ort_root:
        cmake_configure.append(f"-DORT_ROOT={ort_root}")

    _run(cmake_configure, cwd=build_dir)
    _run(["cmake", "--build", ".", "--parallel", str(jobs)], cwd=build_dir)
    print(f"\n[done] build  ({_elapsed(t0)})")


def step_benchmark(skip: bool, build_dir: Path) -> None:
    if skip:
        print("[skip] benchmark skipped (--skip-benchmark)")
        return
    _step("Step 4 / 4 — Inference benchmark")
    t0 = time.perf_counter()
    inference_bin = build_dir / "inference"
    trt_bin = build_dir / "trt_inference"
    _run(
        [
            PY,
            "src/benchmark.py",
            "--onnx-dir",
            str(ONNX_DIR),
            "--inference-bin",
            str(inference_bin),
            "--trt-bin",
            str(trt_bin),
        ]
    )
    print(f"\n[done] benchmark  ({_elapsed(t0)})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: generate → train → build → benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ort-root",
        default=None,
        help=("Path to ONNX Runtime tar.gz install (e.g. ~/onnxruntime). Omit if ORT was installed via apt (libonnxruntime-dev)."),
    )
    p.add_argument(
        "--build-dir",
        type=Path,
        default=REPO_ROOT / "cpp" / "build",
        help="C++ build directory (default: cpp/build)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 4,
        help="Parallel build jobs (default: all cores)",
    )
    p.add_argument("--skip-generate", action="store_true", help="Skip data generation if CSV exists")
    p.add_argument("--skip-train", action="store_true", help="Skip training if .pth artifacts exist")
    p.add_argument("--skip-build", action="store_true", help="Skip C++ binary build")
    p.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ort_root = _resolve_ort_root(args.ort_root)

    # Line-buffer stdout so pipeline headers appear before subprocess output
    # even when stdout is piped/captured (non-tty).
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    wall_start = time.perf_counter()

    ort_label = ort_root if ort_root else "system (apt deb)"
    print(f"[pipeline] repo root : {REPO_ROOT}")
    print(f"[pipeline] ORT       : {ort_label}")
    print(f"[pipeline] build dir : {args.build_dir}")
    print(f"[pipeline] jobs      : {args.jobs}")

    step_generate(skip=args.skip_generate)
    step_train(skip=args.skip_train)
    step_build(
        skip=args.skip_build,
        ort_root=ort_root,
        build_dir=args.build_dir,
        jobs=args.jobs,
    )
    step_benchmark(skip=args.skip_benchmark, build_dir=args.build_dir)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete  ({_elapsed(wall_start)} total)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
