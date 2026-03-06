# foundcog_analysis

Preprocessing and Analysis Pipelines for the FOUNDCOG Project.

## Overview

This repository contains NiPype-based pipelines for preprocessing and GLM analysis of fMRI data from the FOUNDCOG dataset. The dataset follows BIDS convention. Pipelines are designed to run on a SLURM cluster but can also be run single-threaded locally.

---

## Repository Structure

```
foundcog_analysis/
├── foundcog_preproc.py        # Entry point: subject-level preprocessing pipeline
├── foundcog_glm.py            # Entry point: subject-level GLM pipeline
├── preproc/
│   ├── bold_preproc.py        # NiPype sub-workflow for BOLD preprocessing
│   └── reference_files/
│       ├── flirt_in_matrix.mat              # Initial affine matrix for FLIRT normalisation
│       └── flirt_dof_manualselection.json   # Per-subject manual DOF selections for FLIRT
└── analysis/
    └── glm.py                 # NiPype interfaces for the GLM pipeline
```

---

## Pipeline Scripts

### `foundcog_preproc.py` — Preprocessing Pipeline

Builds and runs a NiPype workflow that preprocesses BOLD fMRI data for each subject/session/run. Configure the variables at the top of the file before running.

**Steps per run:**
1. BOLD-specific preprocessing: motion correction via MCFLIRT, framewise displacement estimation
2. Fieldmap-based distortion correction (FSL `topup` / `applytopup`) — applied per session when fieldmaps are available
3. Co-registration of runs to a within-subject mean reference (FSL `FLIRT`)
4. Normalisation to an age-appropriate NIHPD template (FSL `FLIRT`, tested at DOF 6/9/12)
5. Spatial smoothing (8mm FWHM) and ROI timecourse extraction (Schaefer 400-parcel atlas)

**Key configuration constants** (top of file):

| Constant | Description |
|---|---|
| `SINGLE_THREADED` | Set `True` to run locally without SLURM |
| `experiment_dir` | Root of the BIDS dataset |
| `TEMPLATES_DIR` | Root directory for age-appropriate templates |
| `TEMPLATE_MAP` | Masked template per cohort, used as FLIRT reference |
| `TEMPLATE_NORM_2MM_MAP` | 2mm template per cohort, used for `ApplyXFM` |
| `SCHAEFER_ROI_MAP` | Schaefer parcellation per cohort for ROI extraction |

**Outputs** (written to `<experiment_dir>/derivatives/foundcog_preproc/`):

Per run (always):
- `motion_parameters` — raw MCFLIRT `.par` files
- `motion_fwd` — framewise displacement timecourse
- `motion_plots` — rotation and translation plots
- `bad_volumes` — axial and sagittal QA images of the highest-FWD volumes
- `bad_volumes_fwd` — FWD values for those volumes
- `snr` — tSNR map in subject space (mean / std across time)

Per DOF (6, 9, 12 — always):
- `submean_affineflirt` — within-subject mean image normalised to template at each DOF
- `submean_affineflirt_matrix` — corresponding FLIRT transformation matrix
- `submean_affineflirt_figure` — QA overlay figure

Per run (only once a DOF has been manually selected for the subject):
- `normalized_to_common_space` — spatially normalised BOLD time series
- `snr_normalized_to_common_space` — tSNR map normalised to template space
- `smoothing` — spatially smoothed normalised BOLD (8mm FWHM)
- `roi_extraction` — Schaefer 400-parcel ROI timecourses
- `globals` — global signal timecourse

Per subject (only once a DOF has been manually selected):
- `submean_affineflirt_manualselection` — within-subject mean normalised at the selected DOF
- `submean_affineflirt_matrix_manualselection` — corresponding FLIRT matrix
- `submean_affineflirt_figure_manualselection` — QA overlay figure

**Notes:**
- The script reads `preproc/reference_files/flirt_dof_manualselection.json` to apply a manually chosen DOF per subject. Subjects not yet in this file will be added automatically with an empty entry; the full normalisation and smoothing steps are skipped until a DOF is selected.
- One NiPype workflow per subject is submitted to SLURM via the `SLURMGraph` plugin.

---

### `foundcog_glm.py` — GLM Pipeline

Builds and runs a NiPype workflow that fits a first-level GLM on preprocessed BOLD data for the pictures task. Configure the block labelled `### Configuration ###` before running.

**Steps per run:**
1. Resolve file paths for functional images, events, motion parameters, and framewise displacement
2. Build a design matrix using the chosen stimulus design strategy
3. Fit a first-level GLM (nilearn `FirstLevelModel`)
4. Extract per-condition beta coefficients for downstream MVPA

**Key configuration constants** (top of file):

| Constant | Description |
|---|---|
| `SINGLE_THREADED` | Set `True` to run locally without SLURM |
| `EXPERIMENT_DIR` | Root of the BIDS dataset |
| `TEMPLATES_DIR` | Root directory for templates |
| `BRAIN_MASK_MAP` | Brain mask per cohort (keyed by first character of subject ID) |
| `TR` | Repetition time in seconds |
| `FWD_CUTOFF` | Framewise displacement threshold for motion censoring (mm) |
| `FUNC_DERIVATIVES` | Which preprocessed derivative to use as input |
| `EXEMPLAR` | Model each stimulus exemplar as a separate condition |
| `REPETITIONS` | Mark repeated stimulus presentations |

**Outputs** (written to `<experiment_dir>/derivatives/foundcog_glm/`):
- Per-condition beta coefficient maps

---

## Library Packages

### `preproc/bold_preproc.py`

Defines `get_wf_bold_preproc()`, which returns a NiPype sub-workflow used by `foundcog_preproc.py`. It handles:
- Reference volume extraction (middle volume)
- Motion correction (`MCFLIRT`)
- Motion parameter normalisation and framewise displacement estimation

### `analysis/glm.py`

Defines NiPype-compatible `BaseInterface` classes used by `foundcog_glm.py`:

| Class | Role |
|---|---|
| `GLMExperimentSetter` | Resolves file paths and conditions for a given subject/session/run/task |
| `GLMDesign` | Builds the first-level design matrix |
| `GLMRun` | Fits the GLM and computes contrast maps |
| `GLMBetas` | Extracts per-condition beta coefficient images |

---

## Setup

Install the package in editable mode so that the `preproc` and `analysis` packages are importable:

```bash
pip install -e .
```

Key dependencies: `nipype`, `nilearn`, `niworkflows`, `pybids`, `nibabel`, `numpy`, `pandas`, `fsl` (must be available on `PATH`), `ants`.

---

## Age Cohorts and Templates

Subject IDs encode cohort membership:
- **2XXX** — younger infants (~2–5 months); uses the NIHPD 02–05 month template
- **9XXX** — older infants (~8–11 months); uses the NIHPD 08–11 month template

Template source: https://www.mcgill.ca/bic/software/tools-data-analysis/anatomical-mri/atlases/nihpd2
