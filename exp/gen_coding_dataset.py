#!/usr/bin/env python3
"""Repository script-driven coding dataset pipeline.

Follows README pipeline order by orchestrating existing scripts:
1) scripts/magpie_code.sh (batched SFT generation)
2) scripts/magpie-multi-turn.sh (optional)
3) scripts/unitag.sh (quality/classification/language)
4) exp/gen_dis.py (repetition distance)
5) Final deterministic filter + CSV export
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
EXP_DIR = REPO_ROOT / "exp"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


@dataclass
class GPUInfo:
    index: int
    free_mb: int


@dataclass
class Config:
    num_samples: int
    mode: str
    output_csv: str
    model_path: str = DEFAULT_MODEL
    languages: Optional[List[str]] = None

    # generation knobs used by scripts/magpie_code.sh
    ins_topp: float = 1.0
    ins_temp: float = 1.0
    res_topp: float = 1.0
    res_temp: float = 0.0

    # multi-turn and filtering
    mt_turns: int = 3
    oversample_factor: float = 1.8
    quality_allow: Optional[List[str]] = None
    min_neighbor_distance: float = 0.05
    max_repeat_count: int = 0

    # runtime
    device: Optional[str] = None  # auto if None
    distance_device: Optional[str] = None

    # gen_dis params
    search_space_size: int = 500
    search_batch_size: int = 1024
    encoding_batch_size: int = 32768


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def detect_gpus() -> List[GPUInfo]:
    try:
        lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
        ).strip().splitlines()
    except Exception:
        return []
    gpus = []
    for line in lines:
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2:
            gpus.append(GPUInfo(index=int(p[0]), free_mb=int(p[1])))
    return sorted(gpus, key=lambda x: x.free_mb, reverse=True)


def auto_device(cfg: Config) -> tuple[str, str]:
    if cfg.device:
        first = cfg.device.split(",")[0]
        return cfg.device, (cfg.distance_device or first)
    gpus = detect_gpus()
    if not gpus:
        raise RuntimeError("No NVIDIA GPUs found. Provide --device manually on a GPU host.")
    device = ",".join(str(g.index) for g in gpus)
    return device, (cfg.distance_device or str(gpus[0].index))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device, self.distance_device = auto_device(cfg)
        print(f"[runtime] generation_devices={self.device} distance_device={self.distance_device}")

    @property
    def total_prompts(self) -> int:
        return int(self.cfg.num_samples * self.cfg.oversample_factor)

    def _find_latest_job_dir(self, before: set[str]) -> Path:
        after = {p.name for p in DATA_DIR.iterdir() if p.is_dir()}
        new_dirs = sorted(after - before)
        if new_dirs:
            return DATA_DIR / new_dirs[-1]
        # fallback: newest dir
        dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()], key=lambda x: x.stat().st_mtime)
        if not dirs:
            raise RuntimeError("No job directory found under data/")
        return dirs[-1]

    def stage_generate_single_turn(self) -> Path:
        before = {p.name for p in DATA_DIR.iterdir() if p.is_dir()} if DATA_DIR.exists() else set()
        cmd = [
            "bash",
            "magpie_code.sh",
            self.cfg.model_path,
            str(self.total_prompts),
            str(self.cfg.ins_topp),
            str(self.cfg.ins_temp),
            str(self.cfg.res_topp),
            str(self.cfg.res_temp),
        ]
        env = dict(**__import__("os").environ)
        env["CUDA_VISIBLE_DEVICES"] = self.device
        subprocess.run(cmd, cwd=str(SCRIPTS_DIR), check=True, env=env)

        job_dir = self._find_latest_job_dir(before)
        candidates = sorted(job_dir.glob("*_ins_res.json"), key=lambda x: x.stat().st_mtime)
        if not candidates:
            raise RuntimeError(f"No *_ins_res.json produced in {job_dir}")
        return candidates[-1]

    def stage_multi_turn(self, input_file: Path) -> Path:
        if self.cfg.mode != "multi-turn":
            return input_file
        run_cmd(
            [
                "bash",
                "magpie-multi-turn.sh",
                str(input_file),
                str(self.cfg.mt_turns),
                self.device,
                self.cfg.model_path,
            ],
            cwd=SCRIPTS_DIR,
        )
        out = input_file.with_name(input_file.stem + "_mt.json")
        if not out.exists():
            raise RuntimeError(f"Expected multi-turn file missing: {out}")
        return out

    def stage_tagging(self, input_file: Path) -> Path:
        current = input_file
        for mission, suffix in [("quality", "quality"), ("classification", "category"), ("language", "language")]:
            run_cmd(
                [
                    "bash",
                    "unitag.sh",
                    str(current),
                    mission,
                    self.distance_device,
                    self.cfg.model_path,
                ],
                cwd=SCRIPTS_DIR,
            )
            current = current.with_name(current.stem + f"_{suffix}.json")
            if not current.exists():
                raise RuntimeError(f"Expected tagged file missing: {current}")
        return current

    def stage_distance(self, tagged_file: Path) -> Path:
        run_cmd(
            [
                sys.executable,
                "gen_dis.py",
                "--input_file",
                str(tagged_file),
                "--encoding_batch_size",
                str(self.cfg.encoding_batch_size),
                "--distance_distance_threshold",
                str(self.cfg.min_neighbor_distance),
                "--search_space_size",
                str(self.cfg.search_space_size),
                "--search_batch_size",
                str(self.cfg.search_batch_size),
                "--device",
                self.distance_device,
            ],
            cwd=EXP_DIR,
        )
        return DATA_DIR / f"{tagged_file.stem}_distance.jsonl"

    def stage_filter_export(self, distance_file: Path) -> None:
        rows = load_jsonl(distance_file)
        quality_allow = set(self.cfg.quality_allow or ["good", "excellent"])

        filtered: List[Dict[str, Any]] = []
        for row in rows:
            if row.get("input_quality") not in quality_allow:
                continue
            if row.get("task_category") not in {"Coding & Debugging", "Data analysis", "Reasoning"}:
                continue
            if float(row.get("min_neighbor_distance", 0.0)) < self.cfg.min_neighbor_distance:
                continue
            if int(row.get("repeat_count", 0)) > self.cfg.max_repeat_count:
                continue
            if self.cfg.languages and row.get("language") not in self.cfg.languages:
                continue
            filtered.append(row)
            if len(filtered) >= self.cfg.num_samples:
                break

        if not filtered:
            raise RuntimeError("No samples after filters. Increase oversample or relax thresholds.")

        out = Path(self.cfg.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "id", "pre_query_template", "instruction", "response", "created",
            "gen_input_configs", "gen_response_configs", "task_category",
            "other_task_category", "input_quality", "quality_explanation", "language",
            "min_neighbor_distance", "repeat_count", "min_similar_conversation_id",
        ]
        for k in sorted(filtered[0].keys()):
            if k.startswith("instruction_") or k.startswith("response_"):
                columns.append(k)

        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for i, r in enumerate(filtered):
                r["id"] = i
                w.writerow({c: r.get(c) for c in columns})
        print(f"[done] exported {len(filtered)} rows to {out}")

    def run(self) -> None:
        ins_res = self.stage_generate_single_turn()
        stage_base = self.stage_multi_turn(ins_res)
        tagged = self.stage_tagging(stage_base)
        distance = self.stage_distance(tagged)
        self.stage_filter_export(distance)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Custom coding dataset pipeline built on Magpie scripts")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--mode", choices=["single-turn", "multi-turn"], default="single-turn")
    p.add_argument("--output-csv", type=str, default="./data/magpie_coding_dataset.csv")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--languages", type=str, default="python,javascript,java,cpp,go,rust,typescript")
    p.add_argument("--ins-topp", type=float, default=1.0)
    p.add_argument("--ins-temp", type=float, default=1.0)
    p.add_argument("--res-topp", type=float, default=1.0)
    p.add_argument("--res-temp", type=float, default=0.0)
    p.add_argument("--mt-turns", type=int, default=3)
    p.add_argument("--oversample-factor", type=float, default=1.8)
    p.add_argument("--quality-allow", type=str, default="good,excellent")
    p.add_argument("--min-neighbor-distance", type=float, default=0.05)
    p.add_argument("--max-repeat-count", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="Manual generation devices, e.g. 0 or 0,1")
    p.add_argument("--distance-device", type=str, default=None, help="Manual device for unitag/gen_dis")
    p.add_argument("--search-space-size", type=int, default=500)
    p.add_argument("--search-batch-size", type=int, default=1024)
    p.add_argument("--encoding-batch-size", type=int, default=32768)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    if args.config:
        return Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))

    return Config(
        num_samples=args.num_samples,
        mode=args.mode,
        output_csv=args.output_csv,
        model_path=args.model_path,
        languages=[x.strip().lower() for x in args.languages.split(",") if x.strip()],
        ins_topp=args.ins_topp,
        ins_temp=args.ins_temp,
        res_topp=args.res_topp,
        res_temp=args.res_temp,
        mt_turns=args.mt_turns,
        oversample_factor=args.oversample_factor,
        quality_allow=[x.strip() for x in args.quality_allow.split(",") if x.strip()],
        min_neighbor_distance=args.min_neighbor_distance,
        max_repeat_count=args.max_repeat_count,
        device=args.device,
        distance_device=args.distance_device,
        search_space_size=args.search_space_size,
        search_batch_size=args.search_batch_size,
        encoding_batch_size=args.encoding_batch_size,
    )


def main() -> None:
    cfg = build_config(parse_args())
    if "8b" not in cfg.model_path.lower() or "llama" not in cfg.model_path.lower():
        raise ValueError("Model requirement: use a Llama 8B model for generation.")
    Pipeline(cfg).run()


if __name__ == "__main__":
    main()
