# Running Experiments: Drift Types and SAGE Validation

This guide explains how to run the main experiments and reproduce the results step by step.

---

## Prerequisites

### 1. Environment

From the project root:

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Optional packages** (used by `run_sage_validation.py`):

- `sage-importance` – required for SAGE validation
- `POT` – for multivariate Wasserstein in D_het (`pip install POT`)
- `goodpoints` – for compress++ coresets; if missing, K-means fallback is used
- `joblib` – for parallel client SAGE (`--n-jobs`)

### 2. Running on a cluster (Slurm)

If you use Slurm, prefix commands with `srun` (or submit a job script):

```bash
srun python run_drift_types.py --experiments fed_heart_sudden --seeds 42
srun python run_sage_validation.py --dataset hyperplane
```

---

## Part 1: Drift-type experiments (`run_drift_types.py`)

These experiments train federated models under different drift settings (sudden, gradual), run drift detection (RDS, CUSUM, ADWIN, etc.), and optionally run SAGE-based diagnosis.

### What gets run

- **Experiments** are defined in `EXPERIMENTS` in `run_drift_types.py` (e.g. `fed_heart_sudden`, `diabetes_gradual`, `hyperplane_gradual`, `wine_sudden`, `credit_sudden`, `agrawal_sudden`, etc.).
- **Output:** one directory per run:  
  `results/drift_types/<experiment_name>_seed<seed>/`  
  containing `loss_matrix.npy`, `metadata.json`, `trigger_results.json`, `timings.csv`, plots, and (if diagnosis is run) diagnosis outputs.

### Step-by-step: Drift-type experiments

**Step 1 – Run all experiments (all seeds)**

```bash
python run_drift_types.py
```

This uses the default seed `42`. To run multiple seeds:

```bash
python run_drift_types.py
```

**Step 2 – Run only specific experiments**

```bash
# Single experiment
python run_drift_types.py --experiments fed_heart_sudden

# Multiple experiments
python run_drift_types.py --experiments fed_heart_sudden diabetes_gradual hyperplane_gradual
```

**Step 3 – Single seed (shorthand)**

```bash
python run_drift_types.py --seed 42
```

**Step 4 – Quick sanity check (tiny mode)**

Reduces rounds and warmup so the pipeline runs in a few minutes:

```bash
python run_drift_types.py --tiny
```

**Step 5 – Re-run detection only (no training)**

If you changed detection code or thresholds and want to re-run only detection on existing loss data:

```bash
python run_drift_types.py --redetect
```

**Step 6 – Diagnosis only (on an existing run)**

Run diagnosis on a single run directory (no training):

```bash
python run_drift_types.py --diagnose-only fed_heart_sudden_seed42
```

(Use the run directory name as it appears under `results/drift_types/`.)

---

### Drift-types: Post-processing and figures

After runs exist under `results/drift_types/`, use these scripts from the project root.

**Step A – Aggregate results and build summary tables**

```bash
python aggregate_drift_types_summary.py
```

- **Input:** `results/drift_types/*_seed*/` (with `trigger_results.json`, etc.).
- **Output (in `results/drift_types/summary/`):**
  - `detection_table.csv`, `detection_table.tex`
  - `loss_panel.png`, `loss_panel.pdf`

Override paths if needed:

```bash
python aggregate_drift_types_summary.py --results-dir results/drift_types
```

**Step B – Loss + RDS panel (4 experiments)**

```bash
python plot_loss_panel_adhoc.py
```

- **Output:** `results/drift_types/summary/loss_panel.png` (and `.pdf`).
- Optional: `--results-dir`, `--out-dir`.

**Step C – Runtime panel**

```bash
python plot_runtime_panel.py
```

- **Output:** `results/drift_types/summary/runtime_panel.png` (and `.pdf`).

**Step D – RDS hyperparameter sweep (and appendix panel)**

```bash
python sweep_rds_hyperparams.py --results-dir results/drift_types --out-dir results/drift_types/summary
```

- **Input:** existing `loss_matrix.npy` and `metadata.json` under `results/drift_types/*_seed*/`.
- **Output (in `results/drift_types/summary/`):**
  - `rds_sweep_fwer.png`, `rds_sweep_window.png`, `rds_sweep_consecutive.png`
  - `rds_sweep_panel.png` (combined appendix panel)
  - `rds_sweep_results.csv`

---

### Full drift-types pipeline (summary)

```bash
# 1) Run experiments 
python run_drift_types.py --seeds 42 

# 2) Aggregate and build tables + loss/fi panels
python aggregate_drift_types_summary.py

# 3) Optional: loss panel (if not already from step 2)
python plot_loss_panel_adhoc.py

# 4) Runtime panel
python plot_runtime_panel.py

# 5) RDS sweep and appendix panel
python sweep_rds_hyperparams.py --results-dir results/drift_types --out-dir results/drift_types/summary
```

---

## Part 2: SAGE validation experiments (`run_sage_validation.py`)

These experiments compare federated SAGE (IID and compressed) to centralized SAGE and produce summary tables and plots.

### What gets run

- **Datasets (all available):** `hyperplane`, `wine`, `diabetes`, `credit`, `fed_heart`, `agrawal`.  
- **Configs per dataset:** e.g. `iid`(drift level).
- **Seeds:** default `[42, 43, 44, 45, 46]`.
- **Output:** `results/sage_validation/<dataset>_<config>_seed<seed>/`  
  with `result.json`, `fi_scores.json`, `global_model_final.pt`, etc.



**Step 2 – Run only one dataset**

Use one of: `hyperplane`, `wine`, `diabetes`, `credit`, `fed_heart`, `agrawal`.

```bash
python run_sage_validation.py --dataset hyperplane
```

**Step 3 – Run only one config (e.g. IID)**

```bash
python run_sage_validation.py --config iid
```

**Step 4 – Run a single seed**

```bash
python run_sage_validation.py --seed 42
```

**Step 5 – Skip runs that already have results**

```bash
python run_sage_validation.py --skip-existing
```

**Step 6 – Fast pipeline test (tiny mode)**

```bash
python run_sage_validation.py --tiny
```

**Step 7 – Generate summary tables only (no training)**

After some runs exist:

```bash
python run_sage_validation.py --summarize
```

- **Output (in `results/sage_validation/summary/`):**
  - `sage_validation_summary.csv`, `sage_validation_summary_iid.csv`
  - `sage_validation_table.tex`, `sage_validation_table_iid.tex`

**Step 8 – Parallel SAGE per client**

```bash
python run_sage_validation.py --n-jobs 4
```

---

### SAGE validation: Plots

**Main plots (from IID summary)**

Requires `results/sage_validation/summary/sage_validation_summary_iid.csv` (run experiments with `--config iid` then `--summarize`, or use existing summary).

```bash
python scripts/plot_sage_validation.py
```

- **Output (in `results/sage_validation/summary/`):**
  - `sage_iid_mae_by_dataset.png`, `sage_iid_runtime_by_dataset.png`
  - `sage_iid_spearman_by_dataset.png`, `sage_iid_mae_and_runtime_panel.png` (and `.pdf`)


```bash
python scripts/plot_agrawal_dhet_mae.py
```

- **Output:** `results/sage_validation/summary/agrawal_dhet_vs_mae.png` (and `.pdf`).

---

### Full SAGE validation pipeline (summary)

```bash
# 1) Run IID experiments (or full: remove --config iid)
python run_sage_validation.py --config iid

# 2) Build summary and LaTeX tables
python run_sage_validation.py --summarize

# 3) Generate main plots (MAE, runtime, Spearman, panel)
python scripts/plot_sage_validation.py

# 4) Optional: Agrawal D_het vs MAE (need Agrawal runs for iid/low/mid/high)
python scripts/plot_agrawal_dhet_mae.py
```

---

## Reference: Main script options

### `run_drift_types.py`

| Option | Description |
|--------|--------------|
| `--experiments NAME ...` | Run only these experiments (default: all) |
| `--seed N` | Single seed (e.g. 42) |
| `--seeds N1 N2 ...` | List of seeds |
| `--redetect` | Re-run detection only (no training) |
| `--diagnose-only DIR` | Run diagnosis only on run `DIR` (e.g. `fed_heart_sudden_seed42`) |
| `--tiny` | Quick test: fewer rounds, fixed t0 |

### `run_sage_validation.py`

| Option | Description |
|--------|--------------|
| `--dataset NAME` | Run only this dataset: `hyperplane`, `wine`, `diabetes`, `credit`, `fed_heart`, `agrawal` |
| `--config TAG` | Run only this config (e.g. iid, low, mid, high) |
| `--seed N` | Run only this seed |
| `--summarize` | Only generate summary from existing results |
| `--skip-existing` | Skip runs that already have `result.json` |
| `--tiny` | Fast test: 5 rounds, 1 seed, 1 config per dataset |
| `--n-jobs N` | Parallel workers for per-client SAGE (default: 1) |

---

For more detail on SAGE validation reproduction (metrics, table generation, and plot sources), see `docs/REPRODUCE_SAGE_VALIDATION.md`.
