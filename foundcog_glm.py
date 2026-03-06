"""
foundcog_glm.py — Subject-level GLM pipeline for the FOUNDCOG dataset (pictures task).

This script builds and runs a NiPype workflow that fits a General Linear Model
for each subject/session/run in the BIDS dataset. For each run it:
  1. Resolves file paths (functional image, events, motion parameters)
  2. Builds a design matrix with the chosen stimulus design strategy
  3. Fits a first-level GLM using nilearn
  4. Extracts per-condition beta coefficients for downstream MVPA

Only the pictures task is supported in this public release.
Outputs are written to <EXPERIMENT_DIR>/derivatives/foundcog_glm/.

Configuration
-------------
Edit the block below labelled "### Configuration ###" before running.
The script submits one NiPype workflow per subject to SLURM via the
SLURMGraph plugin. To run locally set SINGLE_THREADED = True.
"""

from os import path

from bids.layout import BIDSLayout

from nipype import config
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataSink

from analysis.glm import GLMExperimentSetter, GLMDesign, GLMRun, GLMBetas

# config.enable_debug_mode()  # uncomment for verbose NiPype logging

# ============================================================
# ### Configuration ###
# ============================================================

# -- Execution --
SINGLE_THREADED = False  # set True to run locally (no SLURM)

# -- Paths --
# Change EXPERIMENT_DIR to the root of your BIDS dataset
EXPERIMENT_DIR = path.abspath(path.join("/foundcog", "dataset_sharing"))
DATABASE_PATH = path.abspath(path.join(EXPERIMENT_DIR, "bidsdatabase"))
DERIVS_DIR = path.join(
    "derivatives", "foundcog_preproc"
)  # preprocessing derivatives, linked with EXPERIMENT_DIR when reading inputs
OUTPUT_DIR = path.join(
    "derivatives", "foundcog_glm"
)  # GLM outputs, linked with EXPERIMENT_DIR in DataSink

# -- Templates --
# Templates live outside the BIDS dataset root
TEMPLATES_DIR = "/foundcog/templates"

# -- Brain masks (by age cohort) --
# Subject IDs starting with "2" are the younger cohort; "9" are the older cohort.
BRAIN_MASK_MAP = {
    "2": path.join(TEMPLATES_DIR, "mask", "nihpd_asym_02-05_fcgmask_2mm.nii.gz"),  # younger cohort (2XXX)
    "9": path.join(TEMPLATES_DIR, "mask", "nihpd_asym_08-11_fcgmask_2mm.nii.gz"),  # older cohort (9XXX)
}

# -- Acquisition parameters --
TR = 0.610  # repetition time in seconds

# -- Motion censoring --
FWD_CUTOFF = 1.5  # framewise displacement threshold (mm); runs exceeding this are excluded

# -- Functional derivative to use as input --
# "normalized_to_common_space" is the standard choice; "smoothing" is an alternative
FUNC_DERIVATIVES = "normalized_to_common_space"

# -- GLM design options --
EXEMPLAR = True    # mark each stimulus exemplar as a separate condition
REPETITIONS = False  # mark repeated presentations (odd/even scheme)

# -- Subject / task filtering --
EXCLUDE_SUBS = []  # subjects to exclude (e.g. TR/sync issues)
# TODO (pre-release): remove this override to run all subjects
SUBJECT_LIST_OVERRIDE = None  # set to None to run all subjects

EXCLUDE_TASKS = ["rest10", "rest5", "videos"]  # tasks never included in this pipeline

# ============================================================
# ### Pipeline parameter string (used in output directory names) ###
# ============================================================

_exemplar = "_exemplar" if EXEMPLAR else ""
_repetitions = "_repetitions" if REPETITIONS else ""
pipeline_param_str = f"{_exemplar}{_repetitions}"
if pipeline_param_str == "":
    pipeline_param_str = "_default"

# ============================================================
# ### Subject / task discovery ###
# ============================================================

layout = BIDSLayout(EXPERIMENT_DIR, database_path=DATABASE_PATH)

subject_list = layout.get_subjects()
task_list = layout.get_tasks()
session_list = layout.get_sessions()
run_list = layout.get_runs()

# Sort to ensure stable ordering — without this, varying order causes NiPype to rerun jobs
subject_list.sort()
task_list.sort()
session_list.sort()
run_list.sort()

subject_list = [s for s in subject_list if s not in EXCLUDE_SUBS]
if SUBJECT_LIST_OVERRIDE is not None:
    subject_list = SUBJECT_LIST_OVERRIDE

task_list = [t for t in task_list if t not in EXCLUDE_TASKS]

print(f"Subjects: {subject_list}")
print(f"Tasks:    {task_list}")

# ============================================================
# ### Build per-subject run lists ###
# ============================================================

# For each subject, find which (session, task, run) combinations have data on disk.
# NiPype requires all iterables to be the same length and stepped in sync.
iter_items = {}
for sub in subject_list:
    iter_items[sub] = {"ses": [], "task": [], "run": []}
    for ses in session_list:
        for task in task_list:
            for run in run_list:
                sub_expt_paths = GLMExperimentSetter(
                    base_dir=EXPERIMENT_DIR,
                    derivative_dir=DERIVS_DIR,
                    func_deriv=FUNC_DERIVATIVES,
                    sub=sub,
                    task=task,
                    session=ses,
                    run=str(run),
                )
                sub_expt_paths._set_expt_paths()
                func_file = sub_expt_paths.func_file

                info = {"sub": sub, "session": ses, "task": task, "run": run}
                target_file = func_file.format(**info)
                if path.isfile(path.join(EXPERIMENT_DIR, target_file)):
                    iter_items[sub]["ses"].append(str(ses))
                    iter_items[sub]["task"].append(str(task))
                    iter_items[sub]["run"].append(str(run))

    if len(iter_items[sub]["ses"]) == 0:
        iter_items.pop(sub, None)

# ============================================================
# ### Workflow construction and execution (one workflow per subject) ###
# ============================================================

for sub, sub_items in iter_items.items():

    # Select brain mask based on age cohort (2XXX = younger, 9XXX = older)
    brain_mask = BRAIN_MASK_MAP[sub[0]]

    working_dir = path.join("workingdir", pipeline_param_str.lstrip("_"), sub)
    glm_wf = Workflow(name=f"foundcog_glm{pipeline_param_str}")
    glm_wf.base_dir = path.join(EXPERIMENT_DIR, working_dir)

    # infosource_sub iterates over the subject ID and pipeline config string
    infosource_sub = Node(
        IdentityInterface(fields=["sub", "config"]), name="infosource_sub"
    )
    infosource_sub.iterables = [
        ("sub", [sub]),
        ("config", [pipeline_param_str[1:]]),  # strip leading underscore
    ]

    # infosource steps through (session, task, run) in lock-step
    infosource = Node(
        IdentityInterface(fields=["session", "task", "run"]), name="infosource"
    )
    infosource.iterables = [
        ("session", sub_items["ses"]),
        ("task", sub_items["task"]),
        ("run", sub_items["run"]),
    ]
    infosource.synchronize = True  # step through all three lists together
    glm_wf.connect([(infosource_sub, infosource, [("sub", "sub")])])

    # Resolve file paths for this subject/session/task/run
    glm_experiment = Node(GLMExperimentSetter(), name="glm_experiment")
    glm_experiment.inputs.base_dir = EXPERIMENT_DIR
    glm_experiment.inputs.derivative_dir = path.join(EXPERIMENT_DIR, DERIVS_DIR)
    glm_experiment.inputs.motion_param_deriv = "motion_parameters"  # 6 motion params
    glm_experiment.inputs.fwd_deriv = "motion_fwd"  # framewise displacement values
    glm_experiment.inputs.func_deriv = FUNC_DERIVATIVES

    glm_wf.connect(
        [
            (
                infosource,
                glm_experiment,
                [
                    ("sub", "sub"),
                    ("task", "task"),
                    ("session", "session"),
                    ("run", "run"),
                ],
            )
        ]
    )

    # Build design matrix and confound regressors
    glm_design = Node(GLMDesign(), name="glm_design")
    glm_design.inputs.fwd_cutoff = FWD_CUTOFF
    glm_design.inputs.repetition_marking = REPETITIONS
    glm_design.inputs.exemplar_marking = EXEMPLAR

    glm_wf.connect(
        [
            (glm_experiment, glm_design, [("paths", "paths")]),
            (infosource, glm_design, [("sub", "sub"), ("task", "task")]),
        ]
    )

    # Fit first-level GLM
    glm_run = Node(GLMRun(), name="glm_run")
    glm_run.inputs.tr = TR
    glm_run.inputs.brain_mask = brain_mask

    glm_wf.connect(
        [
            (
                glm_design,
                glm_run,
                [
                    ("design_elements_perrun", "design_elements_perrun"),
                    ("design_settings", "design_settings"),
                ],
            ),
            (infosource, glm_run, [("sub", "sub"), ("task", "task")]),
        ]
    )

    # Extract per-condition beta coefficients
    glm_betas = Node(GLMBetas(), name="glm_betas")
    glm_wf.connect(
        [
            (glm_run, glm_betas, [("fit_models_perrun", "fit_models_perrun")]),
            (glm_experiment, glm_betas, [("conditions", "task_conditions")]),
            (infosource, glm_betas, [("sub", "sub"), ("task", "task")]),
        ]
    )

    # Save outputs
    datasink = Node(DataSink(), name="datasink")
    datasink.inputs.base_directory = EXPERIMENT_DIR
    datasink.inputs.container = OUTPUT_DIR
    datasink.inputs.regexp_substitutions = [
        # _config__sub_2001  →  sub-2001
        (r"_config__sub_([\w]+)", r"sub-\1"),
        # _config_repetitions_sub_2001  →  sub-2001/repetitions
        (r"_config_([\w_]+)_sub_([\w]+)", r"sub-\2/\1"),
        # _run_1_session_1_task_videos  →  ses-1_run-1_task-videos
        (r"_run_(\d+)_session_(\d+)_task_([^/]+)", r"ses-\2_run-\1_task-\3"),
        (r"/_", r"/"),
    ]

    glm_wf.connect([(glm_run, datasink, [("fit_models_file", "models")])])
    glm_wf.connect([(glm_betas, datasink, [("betas_file", "betas")])])

    if SINGLE_THREADED:
        glm_wf.run()
    else:
        glm_wf.run(
            plugin="SLURMGraph",
            plugin_args={"dont_resubmit_completed_jobs": False},
        )
