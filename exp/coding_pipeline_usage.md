# Coding Pipeline Usage Guide

## Quick start

From repository root:

```bash
python exp/gen_coding_dataset.py --config configs/coding_pipeline_config.json
```

Or CLI-only:

```bash
python exp/gen_coding_dataset.py \
  --num-samples 50000 \
  --mode single-turn \
  --output-csv ./data/coding_50k.csv \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --languages python,javascript,java,cpp,go,rust
```

The pipeline auto-detects GPUs and auto-tunes runtime params.

---

## 1) Local machine (Linux workstation / desktop)

### Requirements
- NVIDIA GPU + CUDA driver (`nvidia-smi` works)
- Python environment with repo requirements installed
- Hugging Face access for Llama 8B model

### Setup
```bash
git clone https://github.com/magpie-align/magpie.git
cd magpie
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

### Run
```bash
python exp/gen_coding_dataset.py --config configs/coding_pipeline_config.json
```

### Optional manual override (if auto settings too aggressive)
```bash
python exp/gen_coding_dataset.py \
  --num-samples 20000 \
  --device 0 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --response-batch-size 64
```

---

## 2) Kaggle Notebook

### Kaggle notes
- Kaggle usually gives 1 GPU (often T4/P100/L4 depending on environment).
- Disk and session time are limited; prefer smaller batches/checkpoints.

### Notebook setup cells
```bash
!git clone https://github.com/magpie-align/magpie.git
%cd magpie
!pip install -r requirements.txt
```

Authenticate HF (use a Kaggle secret/env var):
```python
import os
os.environ["HF_TOKEN"] = "<your_hf_token>"
```
```bash
!huggingface-cli login --token $HF_TOKEN
```

### Run smaller job first
```bash
!python exp/gen_coding_dataset.py \
  --num-samples 2000 \
  --mode single-turn \
  --output-csv /kaggle/working/coding_2k.csv \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --languages python,javascript,java
```

### Download output
- File will be under `/kaggle/working/` if you set that path.

---

## 3) Multi-GPU server

The script auto-uses all visible GPUs by default. For explicit pinning:

```bash
python exp/gen_coding_dataset.py \
  --num-samples 100000 \
  --device 0,1,2,3 \
  --tensor-parallel-size 4 \
  --output-csv ./data/coding_100k.csv
```

---

## Common tuning tips

- If OOM occurs:
  - lower `--gpu-memory-utilization` (e.g., `0.88`)
  - lower `--n-per-round`
  - lower `--response-batch-size`
  - lower `--encoding-batch-size`
- For stricter quality:
  - `--quality-allow excellent`
  - increase `--min-neighbor-distance`
- For multi-turn datasets:
  - `--mode multi-turn --mt-turns 3`

---

## Outputs

- Intermediate artifacts: `data/coding_<model>_<timestamp>/...`
- Final CSV: path from `--output-csv`
- CSV includes Magpie-style base fields plus optional `instruction_2/response_2/...` columns for multi-turn mode.
