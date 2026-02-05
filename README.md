# QBC Active Learning Selector (HPC Version)

This script (`QBC_active_learning_HPC_version.py`) scans molecular dynamics trajectories with a committee of DeePMD models, ranks frames by model disagreement, and exports only the most uncertain snapshots as VASP-ready `POSCAR` folders plus summary CSV files. It is meant to run on shared clusters where inference is cheap but retraining is deferred.

## Directory Layout
- `QBC_active_learning_HPC_version.py` – main entry point, pure inference/selection (no training or job submission).
- `input.in` – key/value config file consumed by the script.
- `models/` – directory holding the DeePMD frozen graphs that make up the committee (≥2 required).
- `md_pools/` – recursive tree of trajectory files; adjust the glob in `POOL` to match your dataset.
- `qbc_poscars/`, `qbc_XYZ/`, `qbc_outputs/` – auto-created each run for POSCAR bundles, marked XYZ trajectory, and tabular diagnostics.

## Requirements
- Python 3.9+ with `numpy`, `pandas`, `ase`.
- `deepmd-kit` ≥ 2.0 (needed for `deepmd.infer.DeepPot`); load the same module your training jobs use.
- Trajectory files readable by ASE (e.g., `lammps-dump-text`, `extxyz`, `traj`, ...).
- A DeePMD type map covering every element present in your trajectories.

## Quickstart
1. Activate the environment that already has `deepmd-kit`, ASE, NumPy, and pandas.
2. Copy `input.in` and adjust:
   - `MODELS` – comma-separated paths to `graph.pb` files.
   - `POOL` – glob pointing to the MD data (supports `**` for recursion).
   - Thresholds (`THRESH_LOW`, `THRESH_HIGH`) or `TOP_K`, weights, stride, etc.
3. (Optional) Clean previous outputs by deleting `qbc_poscars`, `qbc_XYZ`, `qbc_outputs`; the script also empties these targets at runtime.
4. Run `python QBC_active_learning_HPC_version.py` from this folder. The script logs progress, scoring statistics, and the resolved configuration.
5. Inspect `qbc_outputs/selection.csv` to see which frames were exported and adjust thresholds if needed.

## Configuration Reference (`input.in`)
| Key | Purpose / Notes | Default |
| --- | --- | --- |
| `MODELS` | Comma-separated DeePMD frozen graphs for the committee (≥2). | **required** |
| `POOL` | Glob for trajectory files; accepts recursion via `**`. | **required** |
| `ASE_FORMAT` | ASE reader to use (`lammps-dump-text`, `extxyz`, etc.). `None` lets ASE auto-detect. | `lammps-dump-text` |
| `TYPE_MAP` | Either inline list (`N,O,C`) or path to a text file with one element per line. | **required** |
| `THRESH_LOW`/`THRESH_HIGH` | Score window to keep. Provide both, one, or neither (if using `TOP_K`). | none |
| `TOP_K` | Use rank-based selection when thresholds are omitted. | none |
| `METRIC` | Column used for ranking; currently `score` (weighted sum of force/energy/virial disagreement). | `score` |
| `W_FORCE`, `W_ENE`, `W_VIR` | Contribution of std(force), std(energy), std(virial). | `1.0`, `0.0`, `0.0` |
| `FORCE_REDUCER` | `max` (default) or `mean` aggregation of per-atom std. | `max` |
| `FORCE_STD_MODE` | `mag`, `comp_norm`, or `comp_max` to control how per-atom force spread is computed. | `mag` |
| `SKIP`, `STRIDE` | Subsampling knobs applied before scoring each trajectory. | `0`, `1` |
| `LIMIT` | Total number of frames to score (after skip/stride); `None` processes all. | `None` |
| `OUT_DIR`, `OUT_DIR_XYZ`, `OUT_DIR_RESULTS` | Destination folders; emptied and recreated every run. | `./qbc_poscars`, `./qbc_XYZ`, `./qbc_outputs` |
| `TAG` | Optional suffix appended to exported POSCAR folders. | empty |

## What the Script Produces
- `qbc_outputs/scores.csv` – master table with per-frame metadata (`global_idx`, file origin, `dF`, `dE`, `dV`, score, etc.).
- `qbc_outputs/selection.csv` – ranked subset that passed the thresholds or `TOP_K`.
- `qbc_outputs/score_histogram.csv` and `<metric>_values.npy` – quick diagnostics for plotting score distributions.
- `qbc_outputs/run_summary.json` – machine-readable recap (counts, paths, thresholds used).
- `qbc_poscars/POSCAR_rankXXXXX_gYYYYYYY[_TAG]/POSCAR` – one folder per selected frame for easy VASP submission.
- `qbc_XYZ/marked_selection.xyz` – concatenated EXTXYZ with per-atom fields (`std_f`, `local_score`); the atom with the highest local score is relabeled as `H` to simplify visualization in OVITO/ASE GUI.

## Tips
- Always ensure every atom type present in `md_pools/` appears in `TYPE_MAP`; the script raises an explicit error otherwise.
- `FORCE_STD_MODE=comp_norm` reproduces the DP-GEN definition of force uncertainty, while `mag` is less expensive but often sufficient.
- If your dataset is extremely large, use `SKIP`, `STRIDE`, and `LIMIT` to throttle memory usage while still sampling the space.
- Persist the printed `[CFG]` lines from startup into your job logs; they are invaluable when comparing different selection campaigns.
