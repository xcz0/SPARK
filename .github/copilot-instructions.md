# Copilot instructions (SPARK)

## Project snapshot
- Package: `spark` (Python >=3.12) with PyTorch + Lightning + Hydra configs (see `pyproject.toml`, `configs/`).
- Core idea: decoder-only Transformer with **differential attention masks** (card/deck/global) and **time-decay bias** (see `spark/models/components/attention.py`, `spark/utils/masking.py`).

## Golden workflows (commands)
- Data preprocessing (raw -> processed parquet): `python script/data_processed.py` (expects `${DATA_ROOT}/raw`, writes `${DATA_ROOT}/processed`).
- Training (Hydra):
  - Default: `python script/train.py`
  - Switch config group: `python script/train.py model=width128_depth4 trainer=debug`
  - Override params: `python script/train.py model.d_model=256 optimizer.learning_rate=3e-4`
- Tests: `pytest -q` (unit tests in `tests/`, fixtures in `tests/conftest.py`).

## Runtime/config conventions
- Environment vars used by configs:
  - `DATA_ROOT` for `configs/data/default.yaml` (`raw_dir`, `processed_dir`).
  - `OUTPUT_ROOT` for `configs/trainer/default.yaml` (`default_root_dir`).
- `script/train.py` loads `.env` early (before Hydra resolves), from repo root, with `override=True`.
- Hydra output dirs: `configs/config.yaml` sets `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}`; do not hardcode paths.

## Data/Batch contract (critical)
- Collation happens in `spark/data/collator.py` and must produce these keys:
  - `numerical_features`, `categorical_features`, `time_stamps`, `ordinal_targets`, `duration_targets`
  - masks: `padding_mask` (True=valid), `causal_mask` (lower-tri True), `card_mask`, `deck_mask`, `time_diff`
- Mask semantics are **bool where True means allowed/valid**; conversion to additive `-inf` happens inside attention (`_get_additive_mask`).
- Special IDs:
  - `deck_id=0` is treated as “special/unknown” and excluded from deck masking (`create_same_element_mask(..., special_id=0)`).

## Model boundaries (where to change what)
- Architecture wiring: `spark/models/architecture.py` (`InputLayer` -> `TransformerBlock` stack -> `CORALHead` + `DurationHead`).
- Differential attention + time decay: `spark/models/components/attention.py`.
- Training loop/metrics/loss: `spark/models/module.py` (uses `CombinedLoss`, logs `val/loss` etc.).

## Prediction utilities
- Single-sequence prediction helpers live in `spark/predict.py` (builds masks via `spark/utils/masking.py`).
- Note: `script/predict_on_card.py` is currently empty; prefer `spark/predict.py` APIs.

## Repo habits
- Prefer updating Hydra configs in `configs/` over hardcoding constants in code.
- Keep tensor shapes explicit (docstrings usually specify `(batch, seq_len, ...)`) and preserve `padding_mask` semantics when touching masks/attention.
