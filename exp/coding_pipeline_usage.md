# Coding Pipeline Usage Guide (README-style workflow)

This custom pipeline automates the repo’s documented steps (`magpie_code.sh` -> optional multi-turn -> `unitag.sh` -> `gen_dis.py` -> export).

## Quick run

```bash
python exp/gen_coding_dataset.py --config configs/coding_pipeline_config.json
```

## Equivalent manual flow (what the script automates)

```bash
cd scripts
bash magpie_code.sh meta-llama/Meta-Llama-3-8B-Instruct 100000 1 1 1 0
bash magpie-multi-turn.sh <job_dir>/<file>_ins_res.json 3 0 meta-llama/Meta-Llama-3-8B-Instruct   # optional
bash unitag.sh <input_json> quality 0 meta-llama/Meta-Llama-3-8B-Instruct
bash unitag.sh <input_quality_json> classification 0 meta-llama/Meta-Llama-3-8B-Instruct
bash unitag.sh <input_category_json> language 0 meta-llama/Meta-Llama-3-8B-Instruct
cd ../exp
python gen_dis.py --input_file <input_language_json>
```

---

## Local GPU workstation

```bash
python exp/gen_coding_dataset.py \
  --num-samples 50000 \
  --mode single-turn \
  --output-csv ./data/coding_50k.csv \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --languages python,javascript,java,cpp,go,rust
```

If you need to pin GPU(s):

```bash
python exp/gen_coding_dataset.py --num-samples 10000 --device 0 --distance-device 0
```

---

## Kaggle notebook

Use smaller jobs due to session limits.

```bash
!git clone https://github.com/magpie-align/magpie.git
%cd magpie
!pip install -r requirements.txt
```

Authenticate HF token:

```python
import os
os.environ["HF_TOKEN"] = "<hf_token>"
```

```bash
!huggingface-cli login --token $HF_TOKEN
```

Run:

```bash
!python exp/gen_coding_dataset.py \
  --num-samples 2000 \
  --mode single-turn \
  --output-csv /kaggle/working/coding_2k.csv \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --languages python,javascript,java \
  --device 0 --distance-device 0
```

---

## Multi-GPU server

Auto-detection uses all visible GPUs for generation. Explicit pinning example:

```bash
python exp/gen_coding_dataset.py \
  --num-samples 200000 \
  --device 0,1,2,3 \
  --distance-device 0 \
  --output-csv ./data/coding_200k.csv
```

---

## Tuning tips

- If OOM:
  - reduce `--oversample-factor`
  - lower prompt count per run (`--num-samples`) and run multiple times
  - pin to fewer GPUs with `--device`
- For stricter final dataset quality:
  - `--quality-allow excellent`
  - increase `--min-neighbor-distance`
- For multi-turn:
  - `--mode multi-turn --mt-turns 3`

## Output layout

- Intermediate artifacts: `data/<job_name>/...`
- Distance file: `data/*_distance.jsonl`
- Final CSV: path from `--output-csv`
