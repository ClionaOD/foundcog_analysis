from os import path
import pandas as pd

import sys
# Ensure that the top-level repo dir (which contains 'analysis') is in sys.path
repo_root = path.abspath(path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from bids.layout import BIDSLayout

from nipype import config
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataSink

from analysis.glm import GLMExperimentSetter, GLMDesign, GLMRun, GLMBetas

config.enable_debug_mode()

##############

experiment_dir = path.abspath(path.join("/foundcog", "dataset_sharing"))
database_path = path.abspath(path.join(experiment_dir, "bidsdatabase"))
derivs_dir = path.join(
    "derivatives", "foundcog_preproc") # will be linked with experiment_dir when reading inputs to pipeline
output_dir = path.join(
    "derivatives", "foundcog_glm"
)  # will be linked with experiment_dir in DataSink


FUNC_DERIVATIVES = "normalized_to_common_space"  # which functional images to use. E.g. alternative would be "smoothing"
TR = 0.610
fwd_cutoff = 1.5

layout = BIDSLayout(experiment_dir, database_path=database_path)

# list of subject identifiers
subject_list = layout.get_subjects()
task_list = layout.get_tasks()
session_list = layout.get_sessions()
run_list = layout.get_runs()

# Painful lession - if these aren't sorted, order varies across runs and causes the whole pipeline to rerun
subject_list.sort()
task_list.sort()
session_list.sort()
run_list.sort()

## Exclude 9022 because TRs/syncing messed up
exclude_subs = ["9022"]
subject_list = [i for i in subject_list if not i in exclude_subs]

subject_list.sort()
subject_list = ['2001', '2002']

print(f"Subjects {subject_list}")

## Getting values to iterate over for each subject
iter_items = {}
for sub in subject_list:
    iter_items[sub] = {"ses": [], "task": [], "run": []}
    for ses in session_list:
        for task in task_list:
            for run in run_list:
                sub_expt_paths = GLMExperimentSetter(
                    base_dir=experiment_dir,
                    derivative_dir=derivs_dir,
                    func_deriv=FUNC_DERIVATIVES,
                    sub=sub,
                    task=task,
                    session=ses,
                    run=str(run)
                )
                sub_expt_paths._set_expt_paths()
                func_file = sub_expt_paths.func_file
                
                info = {
                    "sub": sub,
                    "session": ses,
                    "task": task,
                    "run": run,
                }
                target_file = func_file.format(**info)
                if path.isfile(path.join(experiment_dir, target_file)):
                    iter_items[sub]["ses"].append(str(ses))
                    iter_items[sub]["task"].append(str(task))
                    iter_items[sub]["run"].append(str(run))

for sub in subject_list:

    for task in ["pictures"]:
        for reps, eg in zip([True, False], [False, True]):
            if task == "videos" and eg:
                continue

            # NiPype setup
            working_dir = path.join("workingdir", sub)

            glm_wf = Workflow(name="foundcog_glm")
            glm_wf.base_dir = path.join(experiment_dir, working_dir)

            # Infosource - a function free node to iterate over our items
            infosource = Node(
                IdentityInterface(fields=["sub", "task"]), name="infosource"
            )
            infosource.iterables = [("sub", [sub]), ("task", [task])]

            # Firstly get the paths for all files we need. This is experiment specific.
            # Make sure paths in analysis/glm.py are correct
            glm_experiment = Node(GLMExperimentSetter(), name="glm_experiment")
            glm_experiment.inputs.base_dir = experiment_dir
            glm_experiment.inputs.derivative_dir = path.join(experiment_dir, derivs_dir)
            glm_experiment.inputs.motion_param_deriv = "motion_parameters"  # for 6 motion params
            glm_experiment.inputs.fwd_deriv = "motion_fwd"  # for values use in GLM censoring, standard is framewise displacement
            glm_experiment.inputs.func_deriv = FUNC_DERIVATIVES  # which functional images to use. E.g. alternative would be "smoothing"

            glm_wf.connect(
                [(infosource, glm_experiment, [("sub", "sub"), ("task", "task")])]
            )
            # this outputs a dict of paths

            glm_design = Node(GLMDesign(), name="glm_design")
            ## inputs
            glm_design.inputs.fwd_cutoff = fwd_cutoff
            glm_design.inputs.repetition_marking = reps
            glm_design.inputs.exemplar_marking = eg
            ## other options
            # glm_design.inputs.gaze_coding = False
            # glm_design.inputs.video_tag_marking = False
            # glm_design.inputs.video_tag_path = "/home/clionaodoherty/foundcog_pipeline/events_per_movie_longlist_new.pickle"

            glm_wf.connect(
                [
                    (glm_experiment, glm_design, [("paths", "paths")]),
                    (infosource, glm_design, [("sub", "sub"), ("task", "task")]),
                ]
            )

            glm_run = Node(GLMRun(), name="glm_run")
            glm_run.inputs.tr = TR
            glm_run.inputs.brain_mask = (
                "/foundcog/templates/mask/nihpd_asym_02-05_fcgmask_2mm.nii.gz"
            )

            glm_wf.connect(
                [
                    (
                        glm_design,
                        glm_run,
                        [("design_elements_perrun", "design_elements_perrun"),
                         ("design_settings", "design_settings")],
                    ),
                    (infosource, glm_run, [("sub", "sub"), ("task", "task")]),
                ]
            )

            datasink = Node(DataSink(), name="datasink")
            datasink.inputs.base_directory = experiment_dir
            datasink.inputs.container = output_dir
            datasink.inputs.substitutions = [
                ("_sub_", "sub-"),
                ("task_", "task-"),
            ]

            glm_wf.connect([(glm_run, datasink, [("fit_models_file", "models")])])

            glm_betas = Node(
                GLMBetas(), name="glm_betas"
            )
            glm_wf.connect(
                [
                    (glm_run, glm_betas, [("fit_models_perrun", "fit_models_perrun")]),
                    (glm_experiment, glm_betas, [("conditions", "task_conditions")]),
                    (infosource, glm_betas, [("sub", "sub"), ("task", "task")]),
                ]
            )

            glm_wf.connect([(glm_betas, datasink, [("betas_file", "betas")])])

            # glm_wf.run()


            glm_wf.run(
                plugin="SLURMGraph", plugin_args={"dont_resubmit_completed_jobs": False}
            )
