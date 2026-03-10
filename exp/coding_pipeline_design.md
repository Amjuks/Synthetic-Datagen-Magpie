# Coding Dataset Pipeline Design (Script-First, README-Aligned)

This custom pipeline now follows the repository’s documented workflow and composes the **existing shell scripts** as first-class stages.

## Stage mapping to README pipeline

1. **Batched SFT Data Generation**
   - Runs: `scripts/magpie_code.sh`
   - Produces `*_ins.json` then `*_ins_res.json` in `data/<job>/`.

2. **Batched Multi-turn Data Generation (Optional)**
   - Runs: `scripts/magpie-multi-turn.sh ***_ins_res.json <num_turns> <device> <model>`
   - Produces `*_ins_res_mt.json`.

3. **Dataset Filtering - Tagging**
   - Runs: `scripts/unitag.sh` sequentially for `quality`, `classification`, `language`.
   - Produces chained files: `..._quality.json -> ..._category.json -> ..._language.json`.

4. **Removing Repetition**
   - Runs: `exp/gen_dis.py --input_file ..._language.json`.
   - Produces `data/*_distance.jsonl` with neighbor distance/repeat metadata.

5. **Custom filter + CSV export**
   - Applies deterministic filters (quality/category/language/min-distance/max-repeat).
   - Exports exactly `X` rows to CSV in Magpie-compatible schema fields.

## GPU handling and utilization

- Auto-detects GPUs with `nvidia-smi`.
- Uses all visible GPUs by default for generation scripts (`device="0,1,..."`).
- Uses the most-free GPU as default for tagging and distance stages.
- Supports manual overrides via config/CLI (`device`, `distance_device`).

## Why this matches the repo better

- It reuses script entry points users already run manually (`magpie_code.sh`, `magpie-multi-turn.sh`, `unitag.sh`) rather than bypassing them.
- It keeps Magpie’s intermediate artifacts and naming conventions intact.
- It still adds a reusable “custom pipeline” layer for your codegen-specific filtering/output requirements.
