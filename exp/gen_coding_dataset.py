#!/usr/bin/env python3
"""Repository-native coding synthetic data pipeline.

Composes existing Magpie scripts:
- exp/gen_ins.py (instruction generation)
- exp/gen_res.py (response generation)
- exp/gen_mt.py  (optional multi-turn expansion)
- exp/unitag.py  (quality/category/language tagging)
- exp/gen_dis.py (near-duplicate distance estimation)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "exp"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


@dataclass
class GPUInfo:
    index: int
    total_mb: int
    free_mb: int
    util_pct: int


@dataclass
class RuntimePlan:
    device: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    n_per_round: int
    response_batch_size: int
    encoding_batch_size: int
    distance_device: str


@dataclass
class Config:
    num_samples: int
    mode: str
    output_csv: str
    model_path: str = DEFAULT_MODEL
    languages: Optional[List[str]] = None

    # generation
    instruction_temperature: float = 1.0
    instruction_top_p: float = 1.0
    response_temperature: float = 0.0
    response_top_p: float = 1.0
    response_repetition_penalty: float = 1.0
    mt_turns: int = 3
    oversample_factor: float = 1.8

    # filters
    quality_allow: Optional[List[str]] = None
    min_neighbor_distance: float = 0.05
    max_repeat_count: int = 0

    # explicit overrides (optional)
    device: Optional[str] = None
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    n_per_round: Optional[int] = None
    response_batch_size: Optional[int] = None
    encoding_batch_size: Optional[int] = None

    # distance stage
    search_space_size: int = 500
    search_batch_size: int = 1024


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def detect_gpus() -> List[GPUInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
    except Exception:
        return []

    gpus: List[GPUInfo] = []
    for line in out:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        gpus.append(GPUInfo(index=int(parts[0]), total_mb=int(parts[1]), free_mb=int(parts[2]), util_pct=int(parts[3])))
    return gpus


def build_runtime_plan(cfg: Config) -> RuntimePlan:
    gpus = detect_gpus()

    # User explicit overrides take priority.
    if cfg.device:
        tp = cfg.tensor_parallel_size if cfg.tensor_parallel_size is not None else max(1, len(cfg.device.split(",")))
        return RuntimePlan(
            device=cfg.device,
            tensor_parallel_size=tp,
            gpu_memory_utilization=cfg.gpu_memory_utilization or 0.95,
            n_per_round=cfg.n_per_round or 200,
            response_batch_size=cfg.response_batch_size or 128,
            encoding_batch_size=cfg.encoding_batch_size or 32768,
            distance_device=cfg.device.split(",")[0],
        )

    if not gpus:
        raise RuntimeError(
            "No NVIDIA GPUs detected (nvidia-smi unavailable or no CUDA device). "
            "Use --device to set manually if available, or run this pipeline on a GPU machine."
        )

    # Sort by free memory descending for best utilization.
    gpus = sorted(gpus, key=lambda g: g.free_mb, reverse=True)
    selected = gpus

    # For 8B models, 1-2 GPUs are usually sufficient. Use all visible GPUs for throughput by default.
    device = ",".join(str(g.index) for g in selected)
    tp = len(selected)

    max_free = selected[0].free_mb
    min_free = selected[-1].free_mb

    if min_free >= 70_000:
        n_per_round = 512
        response_batch_size = 512
        encoding_batch_size = 131072
    elif min_free >= 40_000:
        n_per_round = 384
        response_batch_size = 256
        encoding_batch_size = 65536
    elif min_free >= 22_000:
        n_per_round = 256
        response_batch_size = 192
        encoding_batch_size = 49152
    else:
        n_per_round = 128
        response_batch_size = 96
        encoding_batch_size = 32768

    # keep some safety margin
    gpu_mem_util = 0.95 if max_free >= 20_000 else 0.90

    return RuntimePlan(
        device=device,
        tensor_parallel_size=tp,
        gpu_memory_utilization=cfg.gpu_memory_utilization or gpu_mem_util,
        n_per_round=cfg.n_per_round or n_per_round,
        response_batch_size=cfg.response_batch_size or response_batch_size,
        encoding_batch_size=cfg.encoding_batch_size or encoding_batch_size,
        distance_device=str(selected[0].index),
    )


class MagpieCodingPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.runtime = build_runtime_plan(cfg)
        self.timestamp = int(time.time())
        self.job_name = f"coding_{cfg.model_path.split('/')[-1]}_{self.timestamp}"
        self.data_dir = REPO_ROOT / "data" / self.job_name
        self.data_dir.mkdir(parents=True, exist_ok=True)

        print("[runtime]", self.runtime)

    @property
    def target_generated(self) -> int:
        return int(self.cfg.num_samples * self.cfg.oversample_factor)

    def _ins_file(self) -> Path:
        return self.data_dir / f"Magpie_{self.cfg.model_path.split('/')[-1]}_{self.target_generated}_{self.timestamp}_ins.json"

    def _res_file(self) -> Path:
        p = self._ins_file()
        return p.with_name(p.stem.replace("_ins", "_ins_res") + ".json")

    def _mt_file(self) -> Path:
        p = self._res_file()
        return p.with_name(p.stem + "_mt.json")

    def _tag_file(self, suffix: str) -> Path:
        base = self._mt_file() if self.cfg.mode == "multi-turn" else self._res_file()
        return base.with_name(base.stem + f"_{suffix}.json")

    def _distance_file(self) -> Path:
        base = self._tag_file("language")
        return REPO_ROOT / "data" / f"{base.stem}_distance.jsonl"

    def stage_generate_instructions(self) -> None:
        run_cmd(
            [
                sys.executable,
                "gen_ins.py",
                "--device",
                self.runtime.device,
                "--model_path",
                self.cfg.model_path,
                "--control_tasks",
                "code",
                "--total_prompts",
                str(self.target_generated),
                "--n",
                str(self.runtime.n_per_round),
                "--temperature",
                str(self.cfg.instruction_temperature),
                "--top_p",
                str(self.cfg.instruction_top_p),
                "--tensor_parallel_size",
                str(self.runtime.tensor_parallel_size),
                "--gpu_memory_utilization",
                str(self.runtime.gpu_memory_utilization),
                "--job_name",
                self.job_name,
                "--timestamp",
                str(self.timestamp),
                "--output_folder",
                "../data",
            ],
            cwd=EXP_DIR,
        )

    def stage_generate_responses(self) -> None:
        run_cmd(
            [
                sys.executable,
                "gen_res.py",
                "--device",
                self.runtime.device,
                "--offline",
                "--engine",
                "vllm",
                "--model_path",
                self.cfg.model_path,
                "--input_file",
                str(self._ins_file()),
                "--batch_size",
                str(self.runtime.response_batch_size),
                "--temperature",
                str(self.cfg.response_temperature),
                "--top_p",
                str(self.cfg.response_top_p),
                "--repetition_penalty",
                str(self.cfg.response_repetition_penalty),
                "--tensor_parallel_size",
                str(self.runtime.tensor_parallel_size),
                "--gpu_memory_utilization",
                str(self.runtime.gpu_memory_utilization),
            ],
            cwd=EXP_DIR,
        )

    def stage_multi_turn(self) -> None:
        if self.cfg.mode != "multi-turn":
            return
        run_cmd(
            [
                sys.executable,
                "gen_mt.py",
                "--device",
                self.runtime.device,
                "--model_path",
                self.cfg.model_path,
                "--input_file",
                str(self._res_file()),
                "--num_turns",
                str(self.cfg.mt_turns),
                "--batch_size",
                str(self.runtime.response_batch_size),
                "--tensor_parallel_size",
                str(self.runtime.tensor_parallel_size),
                "--gpu_memory_utilization",
                str(self.runtime.gpu_memory_utilization),
            ],
            cwd=EXP_DIR,
        )

    def stage_tagging(self) -> None:
        missions = ["quality", "classification", "language"]
        base = self._mt_file() if self.cfg.mode == "multi-turn" else self._res_file()
        current = base
        for mission in missions:
            run_cmd(
                [
                    sys.executable,
                    "unitag.py",
                    "--offline",
                    "--device",
                    self.runtime.distance_device,
                    "--model_path",
                    self.cfg.model_path,
                    "--tag_mission",
                    mission,
                    "--input_file",
                    str(current),
                    "--save_as",
                    "json",
                ],
                cwd=EXP_DIR,
            )
            suffix = "category" if mission == "classification" else mission
            current = current.with_name(current.stem + f"_{suffix}.json")

    def stage_distance(self) -> None:
        tagged = self._tag_file("language")
        run_cmd(
            [
                sys.executable,
                "gen_dis.py",
                "--input_file",
                str(tagged),
                "--encoding_batch_size",
                str(self.runtime.encoding_batch_size),
                "--distance_distance_threshold",
                str(self.cfg.min_neighbor_distance),
                "--search_space_size",
                str(self.cfg.search_space_size),
                "--search_batch_size",
                str(self.cfg.search_batch_size),
                "--device",
                self.runtime.distance_device,
            ],
            cwd=EXP_DIR,
        )

    def stage_filter_and_export(self) -> None:
        rows = load_jsonl(self._distance_file())
        quality_allow = set(self.cfg.quality_allow or ["good", "excellent"])

        filtered = []
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
            raise RuntimeError("No samples remained after filtering. Relax thresholds/oversampling.")

        out = Path(self.cfg.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "id",
            "pre_query_template",
            "instruction",
            "response",
            "created",
            "gen_input_configs",
            "gen_response_configs",
            "task_category",
            "other_task_category",
            "input_quality",
            "quality_explanation",
            "language",
            "min_neighbor_distance",
            "repeat_count",
            "min_similar_conversation_id",
        ]
        for k in sorted(filtered[0].keys()):
            if k.startswith("instruction_") or k.startswith("response_"):
                columns.append(k)

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for idx, row in enumerate(filtered):
                row["id"] = idx
                writer.writerow({c: row.get(c) for c in columns})

        print(f"[done] exported {len(filtered)} rows to {out}")

    def run(self) -> None:
        self.stage_generate_instructions()
        self.stage_generate_responses()
        self.stage_multi_turn()
        self.stage_tagging()
        self.stage_distance()
        self.stage_filter_and_export()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build coding synthetic data via native Magpie pipeline")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--mode", choices=["single-turn", "multi-turn"], default="single-turn")
    p.add_argument("--output-csv", type=str, default="./data/magpie_coding_dataset.csv")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--languages", type=str, default="python,javascript,java,cpp,go,rust,typescript")
    p.add_argument("--oversample-factor", type=float, default=1.8)
    p.add_argument("--mt-turns", type=int, default=3)
    p.add_argument("--quality-allow", type=str, default="good,excellent")
    p.add_argument("--min-neighbor-distance", type=float, default=0.05)
    p.add_argument("--max-repeat-count", type=int, default=0)

    # Optional manual runtime overrides
    p.add_argument("--device", type=str, default=None, help="Manual CUDA devices e.g. '0' or '0,1'. Overrides auto-detect")
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--n-per-round", type=int, default=None)
    p.add_argument("--response-batch-size", type=int, default=None)
    p.add_argument("--encoding-batch-size", type=int, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return Config(**raw)

    return Config(
        num_samples=args.num_samples,
        mode=args.mode,
        output_csv=args.output_csv,
        model_path=args.model_path,
        languages=[x.strip().lower() for x in args.languages.split(",") if x.strip()],
        oversample_factor=args.oversample_factor,
        mt_turns=args.mt_turns,
        quality_allow=[x.strip() for x in args.quality_allow.split(",") if x.strip()],
        min_neighbor_distance=args.min_neighbor_distance,
        max_repeat_count=args.max_repeat_count,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        n_per_round=args.n_per_round,
        response_batch_size=args.response_batch_size,
        encoding_batch_size=args.encoding_batch_size,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    if "8b" not in cfg.model_path.lower() or "llama" not in cfg.model_path.lower():
        raise ValueError("Model requirement: use a Llama 8B model for generation.")
    MagpieCodingPipeline(cfg).run()


if __name__ == "__main__":
    main()
