
# Backbone (VTA)

This is the simple, script-first forecaster for the VTA pipeline. It combines time-series and text streams, supports classifier-free guidance (CFG), and uses a learned statistical alignment layer. It takes as input the GRPO reference CSVs produced by the `grpo` stage.

---

## Where to put your data

- **Raw stock CSVs** (Yahoo Finance format):  
  Place in `backbone/data/stocknet/`  
  Example: `backbone/data/stocknet/AAPL.csv`
- **GRPO reference CSVs**:  
  Place in `backbone/data/grpo_input/<DATASET_LABEL>/<MODEL_NAME>/`  
  Example: `backbone/data/grpo_input/stocknet/Qwen2.5-7B-Instruct/`

---

## Setup
Use the repo’s unified environment (see top-level README). Then from this folder run once:
```bash
python pca.py
```

## Data
- Raw CSVs (Yahoo Finance format) under `backbone/data/stocknet/`.
- GRPO reference CSVs under `backbone/data/grpo_input/<DATASET_LABEL>/<MODEL_NAME>/`.

- **Model checkpoints**:  
  Written to `backbone/checkpoints/`
- **Metrics, predictions, and arrays**:  
  Written to `backbone/results/`

## Demos
Scripts are under `backbone/scripts/`.
- Single stock (~3–5 min): `./scripts/vta_demo_single.sh`
  - Edit: `ROOT_PATH`, `DATA_PATH`, `REF_CSV_DIR`
- Multi stock (~3–4 hours): `./scripts/vta_demo_multi.sh`
  - Edit: `ROOT_PATH`, `REF_CSV_DIR`

## Acknowledgement
- [CALF] (https://github.com/Hank0626/CALF)
