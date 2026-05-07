# StressGait-MM

Multi-modal gait dataset and benchmark for binary cognitive-stress detection,
pairing skeleton video (three calibrated viewpoints) with geophone
floor-vibration recordings.

---

## Repository layout

```
stressgait/
├── config.py               ← Edit this to set data paths & hyperparameters
├── run.py                  ← Main entry point
├── requirements.txt
│
├── data/
│   ├── subject_matcher.py  ← Skeleton ↔ vibration subject matching
│   ├── skeleton_qc.py      ← QC helpers (validity, speed, segmentation)
│   ├── skeleton_manager.py ← 4-stage data-curation pipeline
│   ├── vibration_cache.py  ← CWT strip image pre-loader
│   └── dataset.py          ← PyTorch Dataset + SubjectIDMap
│
├── models/
│   ├── graph.py            ← SpatialGCN, BRSA, TSA
│   ├── vibration_encoder.py← ResNet-18 + GeM + TSA strip encoder
│   ├── gpc_fusion.py       ← GPC Fusion (novel contribution)
│   └── stress_gait.py      ← MultiModalStressGait (fusion switch)
│
├── losses/
│   └── losses.py           ← Focal, SupCon, SISC, LossBundle
│
├── training/
│   ├── pretrain.py         ← Masked-joint SSL pre-training
│   └── trainer.py          ← Supervised training loop
│
├── evaluation/
│   ├── metrics.py          ← evaluate(), subject_level_aucs(), stats tests
│   └── protocols.py        ← P3/P4/P5 splitters + run_with_splits()
│
├── experiments/
│   ├── grid.py             ← Resumable experiment grid
│   └── ablations.py        ← Ablation runner
│
└── analysis/
    ├── tables.py           ← Table 1, threshold sensitivity table
    └── figures.py          ← Phase-attention, speed, pipeline figures
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure data paths

Open **`config.py`** and set `skeleton_root`, `vib_root`, and (optionally)
`subject_xlsx_path` to point at your local copy of the dataset.

### 3. Sanity check (1 run, 5 epochs, ~10 min on GPU)

```bash
python run.py --mode quick
```

### 4. Single-seed first pass (6 configurations)

```bash
python run.py --mode full --single_seed 42
```

### 5. Full 3-seed grid (18 configurations) + ablations

```bash
python run.py --mode full --run_ablations
```

### 6. Add extended evaluation protocols (P3/P4/P5)

```bash
python run.py --mode full --run_extended
```

### 7. Regenerate tables and figures from existing runs

```bash
python run.py --mode full --analysis_only
```

---

## Dataset layout expected on disk

```
<skeleton_root>/
  <view>/             # farside | middle | nearside
    <condition>/      # normal | cog  (alias for 'oral')
      <subject>/
        skeleton.npy  # (T, 33, 3) float32 — MediaPipe landmarks [0,1]
        metadata.json # {"fps": 20, "resolution": [640, 360]}  (optional)

<vib_root>/
  <sensor>/           # channel_1 | channel_2
    <condition>/      # normal | cog
      <subject>/
        *.png         # CWT strip images
```

---

## Key outputs

All outputs land under `outputs/` (configurable via `Config.output_root`):

| Path | Contents |
|---|---|
| `outputs/runs/<tag>/best.pt` | Best checkpoint |
| `outputs/runs/<tag>/preds_test.parquet` | Test-set predictions |
| `outputs/tables/table1.csv` | Main comparison table |
| `outputs/tables/table_threshold_sensitivity.csv` | Supplementary QC table |
| `outputs/figures/fig_phase_attention.pdf` | GPC attention figure |
| `outputs/figures/fig_speed_invariance.pdf` | Speed-invariance figure |
| `outputs/artifacts/filter_diagnostics.json` | QC funnel counters |
| `outputs/artifacts/ssl_state.pt` | Cached SSL weights |

---

## Resuming interrupted runs

Every run saves predictions to `preds_test.parquet` before exiting.
Re-running any command automatically skips completed experiments
(the `skip_if_done` cache), so a multi-session workflow simply means
re-executing the same command in each new session.

---

## Experiment modes

| `--mode` | Configurations | Typical GPU time |
|---|---|---|
| `quick` | 1 run (seed 42, 5 epochs) | ~10 min |
| `lite`  | 1 method × 2 seeds | ~2 h |
| `full`  | 6 methods × 3 seeds | ~33 h |
