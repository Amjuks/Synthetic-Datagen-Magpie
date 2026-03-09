# Coding Dataset Pipeline Design (Repository-Native + Auto GPU Utilization)

This implementation uses existing repository scripts directly so behavior stays aligned with Magpie.

## End-to-end stages

1. **Instruction generation**: `exp/gen_ins.py --control_tasks code`
2. **Response generation**: `exp/gen_res.py`
3. **Optional multi-turn expansion**: `exp/gen_mt.py`
4. **Quality/tagging reuse**: `exp/unitag.py` (`quality -> classification -> language`)
5. **Near-duplicate detection**: `exp/gen_dis.py`
6. **Filtering + CSV export**: implemented in `exp/gen_coding_dataset.py`

## GPU auto-detection and best utilization

`exp/gen_coding_dataset.py` now:
- Detects GPUs automatically through `nvidia-smi`.
- Sorts GPUs by free memory and uses all visible GPUs by default for generation stages.
- Auto-sets:
  - `device` (e.g. `0,1,2`)
  - `tensor_parallel_size`
  - `gpu_memory_utilization`
  - generation throughput knobs (`n_per_round`, `response_batch_size`, `encoding_batch_size`) based on available memory tiers.
- Lets you override everything manually if needed (`--device`, `--tensor-parallel-size`, etc.).

## Diversity strategy

- Uses Magpie code-control mode (`--control_tasks code`) to bias toward coding tasks.
- Supports language whitelist filter for multilingual coding outputs.
- Uses oversampling + filtering to preserve diversity while enforcing quality.

## Scaling strategy

- Oversampling (`oversample_factor`) to compensate for filtering attrition.
- Reuses native batch/checkpoint mechanics from Magpie scripts.
- Final CSV export is deterministic and bounded to exactly `num_samples`.

## Quality and filtering

- Reuses Magpie quality/classification/language tagging from `unitag.py`.
- Reuses Magpie near-neighbor distance logic from `gen_dis.py`.
- Applies deterministic filtering over quality, task category, language, and repetition/distance thresholds.
