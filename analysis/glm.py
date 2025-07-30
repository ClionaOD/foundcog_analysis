from os import path
import glob

import os
import pickle

import numpy as np
import pandas as pd
from nilearn._utils.niimg_conversions import check_niimg
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import get_data

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    traits,
    File,
)


class GLMPathsInputSpec(BaseInterfaceInputSpec):
    base_dir = traits.Str(mandatory=True, desc="Base directory for the analysis")
    func_deriv = traits.Str(mandatory=True, desc="Functional derivative directory")
    sub = traits.Str(mandatory=True, desc="Subject identifier")
    task = traits.Str(mandatory=True, desc="Task identifier")
    motion_param_deriv = traits.Str(
        default_value="motion_parameters",
        usedefault=True,
        desc="Motion parameter derivative directory",
    )
    fwd_deriv = traits.Str(
        default_value="motion_fwd",
        usedefault=True,
        desc="Framewise displacement derivative directory for censoring",
    )
    derivative_dir = traits.Str(
        default_value="derivatives",
        usedefault=True,
        desc="Directory for the derivative files",
    )


class GLMPathsOutputSpec(TraitedSpec):
    paths = traits.Dict(
        desc="Paths to the functional images, events, motion parameters, and FWD files"
    )


class GLMPathSetter(BaseInterface):

    input_spec = GLMPathsInputSpec
    output_spec = GLMPathsOutputSpec

    def _run_interface(self, runtime):
        self._results = {}
        paths = self._get_fnames()

        self._results["paths"] = paths
        return runtime

    def _list_outputs(self):
        return self._results

    def _set_expt_paths(self):
        # Set paths for the experiment
        self.func_file = path.join(
            self.inputs.base_dir,
            self.inputs.derivative_dir,
            self.inputs.func_deriv,
            "sub-{sub}",
            "ses-?_run-00?_task-{task}",
            # TODO: fix for other filenames
            "sub-{sub}_ses-?_task-{task}_dir-AP_run-00?_bold_mcf_corrected_flirt.nii.gz",
        )

        self.event_file = path.join(
            self.inputs.base_dir,
            "sub-{sub}",
            "ses-{sesnum}",
            "func",
            "sub-{sub}_ses-{sesnum}_task-{task}_dir-AP_run-{runnum}_events.tsv",
        )

        self.motion_param_file = path.join(
            self.inputs.base_dir,
            self.inputs.derivative_dir,
            self.inputs.motion_param_deriv,
            "sub-{sub}",
            "ses-{sesnum}_run-{runnum}_task-{task}",
            "sub-{sub}_ses-{sesnum}_task-{task}_dir-AP_run-{runnum}_bold_mcf.nii.par",
        )

        self.fwd_file = path.join(
            self.inputs.base_dir,
            self.inputs.derivative_dir,
            self.inputs.fwd_deriv,
            "sub-{sub}",
            "ses-{sesnum}_run-{runnum}_task-{task}",
            "fd_power_2012.txt",
        )

    def _match_func_to_datafiles(self, funcrunpath):
        info = funcrunpath.split("/")[-1].split("_")
        runnum = [i for i in info if "run" in i][0].split("-")[-1]
        sesnum = [i for i in info if "ses" in i][0].split("-")[-1]

        event_file = self.event_file.format(
            sub=self.inputs.sub, sesnum=sesnum, task=self.inputs.task, runnum=runnum
        )
        assert path.exists(event_file), f"Event file not found: {event_file}"
        motion_param_file = self.motion_param_file.format(
            sub=self.inputs.sub, sesnum=sesnum, task=self.inputs.task, runnum=runnum
        )
        assert path.exists(
            motion_param_file
        ), f"Motion parameter file not found: {motion_param_file}"
        fwd_file = self.fwd_file.format(
            sub=self.inputs.sub, sesnum=sesnum, task=self.inputs.task, runnum=runnum
        )
        assert path.exists(fwd_file), f"FWD file not found: {fwd_file}"
        return {
            "func": [funcrunpath],
            "events": [event_file],
            "motion": [motion_param_file],
            "fwd": [fwd_file],
            "run_order": [runnum],
            "ses_order": [sesnum],
        }

    def _get_fnames(self):
        self._set_expt_paths()
        funcpaths = glob.glob(
            self.func_file.format(sub=self.inputs.sub, task=self.inputs.task)
        )

        if len(funcpaths) == 0:
            # try again without the corrected suffix, maybe there were no fmaps
            self.func_file = self.func_file.replace(
                "_corrected_flirt.nii.gz", "_flirt.nii.gz"
            )
            funcpaths = glob.glob(
                self.func_file.format(sub=self.inputs.sub, task=self.inputs.task)
            )

        if len(funcpaths) == 0:
            raise FileNotFoundError(
                f"No functional images found for subject {self.inputs.sub} and task {self.inputs.task}."
            )

        paths = {}
        for funcrunpath in funcpaths:
            expt_dict_fnames = self._match_func_to_datafiles(funcrunpath)
            for key, value in expt_dict_fnames.items():
                if key in paths:
                    paths[key].extend(value)
                else:
                    paths[key] = value.copy()  # Use copy() to avoid aliasing

        return paths


class GLMDesignInputSpec(BaseInterfaceInputSpec):
    paths = traits.Dict(
        mandatory=True,
        desc="Paths to the files necessary for GLM analysis. This should include all runs for this participant and task.",
    )
    sub = traits.Str(mandatory=True, desc="Subject identifier")
    task = traits.Str(mandatory=True, desc="Task identifier")

    fwd_cutoff = traits.Float(
        default_value=1.5,
        usedefault=True,
        desc="Framewise displacement cutoff for censoring",
    )

    repetition_marking = traits.Bool(
        default_value=False,
        usedefault=True,
        desc="Whether to mark repetitions in the design matrix",
    )
    exemplar_marking = traits.Bool(
        default_value=False,
        usedefault=True,
        desc="Whether to mark individual exemplars in the design matrix",
    )
    gaze_coding = traits.Bool(
        default_value=False,
        usedefault=True,
        desc="Whether to include tags from the MRI camera recordings in the design matrix",
    )
    video_tag_marking = traits.Bool(
        default_value=False,
        usedefault=True,
        desc="Whether to include video tags from stimulus design in the design matrix",
    )
    video_tag_path = traits.Str(
        default_value="events_per_movie_longlist_new.pickle",
        usedefault=True,
        desc="Path to the video tags file for video tag marking",
    )


class GLMDesignOutputSpec(TraitedSpec):
    design_elements_perrun = traits.Dict(desc="Design elements for each run")
    design_settings = traits.Dict(
        desc="Settings used for the design matrix, including task, fwd_cutoff, and marking options"
    )


class GLMDesign(BaseInterface):
    input_spec = GLMDesignInputSpec
    output_spec = GLMDesignOutputSpec

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        self._results = {}

        self._check_path_dict()

        design_settings = {
            'task': self.inputs.task,
            'fwd_cutoff': self.inputs.fwd_cutoff,
            'repetition_marking': self.inputs.repetition_marking,
            'exemplar_marking': self.inputs.exemplar_marking,
            'gaze_coding': self.inputs.gaze_coding,
            'video_tag_marking': self.inputs.video_tag_marking,
        }

        if not (self.inputs.repetition_marking) and not (self.inputs.exemplar_marking):
            print(
                "Neither repetition marking nor exemplar marking is set. No repetitions or exemplars will be marked in the design matrix."
            )

        if (self.inputs.exemplar_marking) and (self.inputs.task != "pictures"):
            raise ValueError(
                "Exemplar marking is only applicable for the 'pictures' task."
            )

        if (self.inputs.video_tag_marking) and (self.inputs.task != "videos"):
            raise ValueError(
                "Video tag marking is only applicable for the 'videos' task."
            )

        nruns = len(self.inputs.paths["events"])

        run_elements = {
            "func_paths": [],
            "events": [],
            "confounds": [],
            "conf_names": [],
            "run_order": [],
            "ses_order": [],
        }
        for runidx in range(nruns):
            run_design_elements = self._get_design_elements(runidx)
            if run_design_elements is None:
                print(f"Skipping run {runidx} due to excessive spikes.")
                continue
            run_events, confounds, conf_names = run_design_elements

            run_elements["func_paths"].append(self.inputs.paths["func"][runidx])
            run_elements["events"].append(run_events)
            run_elements["confounds"].append(confounds)
            run_elements["conf_names"].append(conf_names)
            run_elements["run_order"].append(self.inputs.paths["run_order"][runidx])
            run_elements["ses_order"].append(self.inputs.paths["ses_order"][runidx])

        self._results["design_elements_perrun"] = run_elements
        self._results["design_settings"] = design_settings
        return runtime

    def _get_design_elements(self, runidx):
        # events
        run_event_path = self.inputs.paths["events"][runidx]
        run_events = self._load_events(run_event_path)

        # get motion_params
        condf, conf_names = self._build_confound_matrix(runidx)

        # get censoring
        spike_arr = self._fwd_censoring(runidx, discard_thresh=0.5)
        if spike_arr is None:
            return None  # Too many spikes, skip this run

        conf_names.extend([f"spike_{i}" for i in range(spike_arr.shape[1])])
        confounds = np.concatenate((condf, spike_arr), axis=1)
        confounds = np.nan_to_num(confounds)

        # get gaze coding
        if self.inputs.gaze_coding:
            tags, tag_names = self._get_gaze_events()

            confounds = np.concatenate((confounds, tags), axis=1)
            conf_names.extend(tag_names)

        return run_events, confounds, conf_names

    def _check_path_dict(self):
        if not isinstance(self.inputs.paths, dict):
            raise ValueError("Paths must be a dictionary containing file paths.")

        required_keys = ["func", "events", "motion", "fwd", "run_order", "ses_order"]
        for key in required_keys:
            if key not in self.inputs.paths:
                raise ValueError(f"Missing required key in paths: {key}")

        if (self.inputs.gaze_coding) and ("camera" not in self.inputs.paths):
            raise ValueError("MRI Camera events are required for gaze coding.")

    def _get_gaze_events(self):
        raise NotImplementedError(
            "Gaze coding is not implemented yet. Please implement the _get_gaze_events method. Must read 'camera' paths during GLMPathSetter._get_fnames()"
        )
        # Example code from previous implementation
        tags = pd.read_csv(paths["camera"][runidx], sep="\t")
        tag_names = tags.columns.to_list()
        return tags, tag_names

    def _get_video_tag_events(self, events_df):
        if not path.exists(self.inputs.video_tag_path):
            raise ValueError(
                f"Video tag file not found: {self.inputs.video_tag_path}. Please provide a valid path."
            )

        raise NotImplementedError(
            "Video tag marking is not implemented yet. Please implement the _get_video_tag_events method."
        )

        # below is an example from previous code
        elan_tags = pd.read_pickle(self.inputs.video_tag_path)
        chosen_trials = ["faces"]  # ,'body_parts','scene','tools']
        # chosen_movies = ['dog','moana','minions_supermarket','forest','bathsong','new_orleans']
        elan_tags = {k.replace(".mp4", ""): v for k, v in elan_tags.items()}
        elan_tags = {
            k: df[df["trial_type"].isin(chosen_trials)] for k, df in elan_tags.items()
        }

        elan = []
        for trial in events_df["trial_type"]:
            if trial not in elan_tags.keys():
                elan.append(events_df[events_df["trial_type"] == trial])
                continue
            elans = elan_tags[trial]
            video_onset = np.array(events_df[events_df["trial_type"] == trial]["onset"])
            for onset in video_onset:
                adjusted_elans = elans.copy()
                adjusted_elans.loc[:, "onset"] = adjusted_elans.loc[:, "onset"] + onset
                elan.append(adjusted_elans)
        elan_df = pd.concat(elan, ignore_index=True)
        elan_df.drop(columns=["magnitude"], inplace=True)
        elan_df.sort_values(by=["onset"], inplace=True, ignore_index=True)
        return elan_df

    def _load_events(self, event_path):
        events_df = pd.read_csv(event_path, sep="\t")
        trial_types = events_df["trial_type"].unique()

        if np.any(["." in i for i in trial_types]):
            from pathlib import Path

            events_df["trial_type"] = [Path(i).stem for i in events_df["trial_type"]]
        events_df["trial_type"] = events_df["trial_type"].str.replace("_", "")

        implicit_baseline = (
            "fixation" if "fixation" in trial_types else "attention_getter"
        )
        events_df = events_df[
            events_df["trial_type"] != implicit_baseline
        ]  # remove implicit baseline

        if self.inputs.video_tag_marking:
            events_df = self._get_video_tag_events()

        if self.inputs.exemplar_marking:
            events_df = self._get_exemplar_events(events_df)

        if self.inputs.repetition_marking:
            events_df = self._get_repetition_events(events_df)

        return events_df

    def _get_exemplar_events(self, events_df):
        from pathlib import Path

        individual_stimuli = [Path(i).stem for i in events_df["stim_file"]]
        individual_stimuli = [i.replace("_", "") for i in individual_stimuli]
        events_df["trial_type"] = individual_stimuli

        return events_df

    def _get_repetition_events(self, events_df, design_nreps=2):
        import math
        from collections import defaultdict

        # Count how many times each trial_type appears
        counts = events_df["trial_type"].value_counts()

        # Calculate exemplars per trial type based on expected repetitions
        num_exemplars_per_type = (counts / design_nreps).astype(int)

        # Track how many times we've seen each trial type
        tracker = defaultdict(int)
        marked_tts = []

        for trial in events_df["trial_type"]:
            if ("attention" in trial) or ("AG" in trial):  # warning, hard coded
                # Skip attention getters
                marked_tts.append(trial)
                continue
            tracker[trial] += 1
            exemplar_count = num_exemplars_per_type[trial]
            rep_num = math.ceil(tracker[trial] / exemplar_count)
            marked_tts.append(f"{trial}_rep{rep_num}")

        events_df["trial_type"] = marked_tts

        return events_df

    def _build_confound_matrix(self, runidx):
        # BUILD CONFOUNDS - do this first to check if run passes motion threshold
        condf = pd.read_csv(
            self.inputs.paths["motion"][runidx], header=None, sep="  ", engine="python"
        )
        conf_names = [
            "motion_x",
            "motion_y",
            "motion_z",
            "rotation_x",
            "rotation_y",
            "rotation_z",
        ]

        return condf.values, conf_names

    def _fwd_censoring(self, runidx, discard_thresh=0.5):
        # Construct spike regressors for this participant
        # 1. Read in the FWD file
        fwd = pd.read_csv(self.inputs.paths["fwd"][runidx])

        # 2. Get indices where this is over cutoff
        # #   Be careful of whether setting 1st or 2nd scan of difference
        # THIS IS FIRST SCAN
        above_idxs = fwd.index[
            fwd["FramewiseDisplacement"] > self.inputs.fwd_cutoff
        ].values

        # THIS WOULD BE SECOND SCAN
        # above_idxs = above_idxs + 1

        # 3. Construct matrix with nrows=nscans, ncols=nframes_todrop
        spike_arr = np.zeros((len(fwd) + 1, above_idxs.size))
        spike_arr[above_idxs, np.arange(above_idxs.size)] = 1

        if len(above_idxs) > len(fwd) * discard_thresh:
            return None  # Too many spikes, skip this run
        else:
            return spike_arr


class GLMRunInputSpec(BaseInterfaceInputSpec):
    design_elements_perrun = traits.Dict(
        mandatory=True,
        desc="Design elements for each run, including functional paths, events, confounds, and confound names",
    )
    design_settings = traits.Dict(
        mandatory=False,
        desc="Settings used for the design matrix, including task, fwd_cutoff, and marking options",
    )
    sub = traits.Str(mandatory=True, desc="Subject identifier for the GLM run")
    task = traits.Str(mandatory=True, desc="Task identifier for the GLM run")

    tr = traits.Float(
        mandatory=True, desc="Repetition time (TR) in seconds for the GLM run"
    )
    brain_mask = traits.Str(
        mandatory=True, desc="Path to the brain mask image for the GLM run"
    )

    fit = traits.Bool(
        default_value=True,
        usedefault=True,
        desc="Whether to fit the GLM model. Useful for debugging.",
    )


class GLMRunOutputSpec(TraitedSpec):
    fit_models_perrun = traits.Dict(
        desc="Fitted models for each run, including design matrices and model objects"
    )
    fit_models_file = File(exists=True, desc="Path to the GLM result file")


class GLMRun(BaseInterface):
    input_spec = GLMRunInputSpec
    output_spec = GLMRunOutputSpec

    def _run_interface(self, runtime):

        self._results = {}

        nruns = len(self.inputs.design_elements_perrun["events"])

        fit_models_perrun = {"models": []}
        for runidx in range(nruns):
            run_func_img = self.inputs.design_elements_perrun["func_paths"][runidx]
            run_events = self.inputs.design_elements_perrun["events"][runidx]
            run_confounds = self.inputs.design_elements_perrun["confounds"][runidx]
            run_conf_names = self.inputs.design_elements_perrun["conf_names"][runidx]
            model = self._fit_glm(
                run_func_img, run_events, run_confounds, run_conf_names
            )
            fit_models_perrun["models"].append(model)

        fit_models_perrun["func_paths"] = self.inputs.design_elements_perrun[
            "func_paths"
        ]
        fit_models_perrun["events"] = self.inputs.design_elements_perrun["events"]
        fit_models_perrun["run_order"] = self.inputs.design_elements_perrun["run_order"]
        fit_models_perrun["ses_order"] = self.inputs.design_elements_perrun["ses_order"]
        fit_models_perrun["design_settings"] = self.inputs.design_settings

        self._results["fit_models_perrun"] = fit_models_perrun

        _repetitions = "_repetition" if self.inputs.design_settings.get(
            "repetition_marking", True
        ) else ""
        _exemplars = "_exemplar" if self.inputs.design_settings.get(
            "exemplar_marking", True
        ) else ""
        _gaze = "_gaze" if self.inputs.design_settings.get(
            "gaze_coding", True
        ) else ""
        _video_tags = "_video_tags" if self.inputs.design_settings.get(
            "video_tag_marking", True
        ) else ""

        save_file = path.abspath(
            f"sub-{self.inputs.sub}_task-{self.inputs.task}{_repetitions}{_exemplars}{_gaze}{_video_tags}_models.pickle"
        )
        with open(save_file, "wb") as f:
            pickle.dump(fit_models_perrun, f)

        self._results["fit_models_file"] = save_file

        return runtime

    def _list_outputs(self):
        return self._results

    def _fit_glm(self, func_img, events, confounds, conf_names):

        model = FirstLevelModel(
            t_r=self.inputs.tr,
            mask_img=self.inputs.brain_mask,
            # all other nilearn defaults
        )

        # BUILD FRAME TIMES - code snippet taken from nilearn
        n_scans = get_data(func_img).shape[3]
        start_time = model.slice_time_ref * model.t_r
        end_time = (n_scans - 1 + model.slice_time_ref) * model.t_r
        frame_times = np.linspace(start_time, end_time, n_scans)

        ## MAKE DESIGN MATRIX
        design = make_first_level_design_matrix(
            frame_times,
            events=events,
            hrf_model=model.hrf_model,
            drift_model=model.drift_model,
            high_pass=model.high_pass,
            drift_order=model.drift_order,
            fir_delays=model.fir_delays,
            add_regs=confounds,
            add_reg_names=conf_names,
            min_onset=model.min_onset,
        )

        ## Fit the model
        if self.inputs.fit:
            func_img_loaded = check_niimg(func_img, ensure_ndim=4)
            model.fit(func_img_loaded, design_matrices=design)

        return model


def single_sub_betas(sub, task, subrunpaths, rep_marking, exemplar_marking):

    def _get_betas(model, colind, cov=None):
        """
        Our usual beta extracting code
        """
        import numpy as np

        labels = model.labels_[0]
        regression_result = model.results_[0]
        effect = np.zeros((labels.size))
        vcov = np.zeros((labels.size))

        for label_ in regression_result.keys():
            if cov is None:
                cov = regression_result[label_].cov

            label_mask = labels == label_
            if label_mask.any():
                resl = regression_result[label_].theta[colind]
                vcov[label_mask] = regression_result[label_].vcov(column=colind)
                effect[label_mask] = resl

        return effect, vcov, cov

    def _mvpa_betas(model_path, all_conds):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        """
        Function to extract betas from the fmri models for later use in MVPA
        args:
            model_path - path to the nilearn FirstLevelModel from which to extract betas
            all_conds - an ideal list of conditions (in the design matrix) for which to calculate betas
        
        returns:
            betas - an array(numvert, numconditions) of beta values for the entire brain volume
            vcov - an array(numvert, numconditions) of variance-covariance values for the entire brain volume
        """
        model = pd.read_pickle(model_path)
        if type(model) == dict:
            model = model["model"]
        numvert = len(model.labels_[0])

        # get columns of interest in design
        cols = model.design_matrices_[0].columns

        # get the conditions in this run - must be in our ideal list
        conditions = []
        for c in cols:
            if (
                (c.split("_")[0] in all_conds)
                or (c[:-1] in all_conds)
                or (c.split("_")[0][:-1] in all_conds)
            ):
                conditions.append(c)

        numcond = len(conditions)

        # Putting this straight into two numpy arrays so we don't need to convert later
        vol_betas = np.zeros((numvert, numcond))
        vol_vcov = np.zeros((numvert, numcond))
        cov = None

        beta_labels = []
        for ind, trial_type in enumerate(conditions):
            colind = cols.get_loc(trial_type)
            beta_labels.append(trial_type)
            effect, vcov, cov = _get_betas(model, colind, cov=cov)
            vol_betas[:, ind] = effect
            vol_vcov[:, ind] = vcov

        # Diagnostics - dump cov to figure

        con = np.zeros((1, cov.shape[0]))
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        cov = cov[list(range(numcond + 1)) + [-1], :]
        cov = cov[:, list(range(numcond + 1)) + [-1]]
        im = ax[0].imshow(cov)
        plt.colorbar(im)

        x = np.arange(-2, 2, 0.05)
        a = np.zeros((numcond + 1, len(x)))
        for colind in range(numcond + 1):
            con = np.zeros((1, numcond + 2))
            con[0, colind] = 1
            for qind, q in enumerate(x):
                con[0, -1] = q
                a[colind, qind] = np.dot(con, np.dot(cov, np.transpose(con)))
        ax[1].plot(x, a.T)

        # plt.savefig(f'sub-{sub}_task-{task}_covariance{_eg_tag}{_rep_tag}.png')
        plt.close()

        return (beta_labels, vol_betas, vol_vcov, cov)

    import os

    if len(subrunpaths) == 0:
        return []

    _rep_tag = "_reps" if rep_marking else ""
    _eg_tag = "_eg" if exemplar_marking else ""

    outpaths = []

    beta_path = os.path.abspath(
        f"sub-{sub}_task-{task}_voxelwisebetas_withvcov{_eg_tag}{_rep_tag}.pickle"
    )

    # used to compare to ideal run
    if task == "videos":
        allconds = [
            "bathsong",
            "dog",
            "neworleans",
            "minionssupermarket",
            "forest",
            "moana",
        ]
    elif task == "pictures":
        allconds = [
            "seabird",
            "crab",
            "dishware",
            "food",
            "tree",
            "squirrel",
            "rubberduck",
            "towel",
            "shelves",
            "shoppingcart",
            "cat",
            "fence",
        ]

    # get each run's beta values and conditions modelled for this run
    beta_save = {}
    for run in subrunpaths:
        _colnames, _runbetas, _runvcov, _runcov = _mvpa_betas(run, allconds)
        beta_save[os.path.basename(run)] = {
            "betas": _runbetas,
            "vcov": _runvcov,
            "colnames": _colnames,
            "cov": _runcov,
        }

    if len(beta_save) > 0:
        with open(beta_path, "wb") as f:
            # pickle.dump(beta_save, f)
            pass
        outpaths.append(beta_path)

    return outpaths


if __name__ == "__main__":

    public_dirs = True

    if public_dirs:
        SUB = "2001"
        INP_DIR = "/foundcog/dataset_sharing"
        DERIV_DIR = "derivatives/foundcog_preproc"
    else:
        SUB = "ICC103"
        INP_DIR = "/foundcog/bids"
        DERIV_DIR = "derivatives"

    TASK = "pictures"

    from nipype import Node

    ## SET PATHS
    glm_paths = Node(GLMPathSetter(), name="glm_path_node")
    glm_paths.inputs.base_dir = INP_DIR
    glm_paths.inputs.derivative_dir = DERIV_DIR
    glm_paths.inputs.func_deriv = "normalized_to_common_space"
    glm_paths.inputs.sub = SUB
    glm_paths.inputs.task = TASK

    path_output = glm_paths.run()
    paths = path_output.outputs.paths

    ## GET DESIGN MATRIX
    glm_design = Node(GLMDesign(), name="glm_design_node")
    glm_design.inputs.sub = SUB
    glm_design.inputs.task = TASK

    glm_design.inputs.repetition_marking = False
    glm_design.inputs.exemplar_marking = True
    glm_design.inputs.gaze_coding = False
    glm_design.inputs.video_tag_marking = False

    glm_design.inputs.paths = paths

    design_output = glm_design.run()

    glm_run = Node(GLMRun(), name="glm_run_node")
    glm_run.inputs.sub = SUB
    glm_run.inputs.task = TASK
    glm_run.inputs.tr = 0.610
    glm_run.inputs.brain_mask = (
        "/foundcog/templates/mask/nihpd_asym_02-05_fcgmask_2mm.nii.gz"
    )
    glm_run.inputs.design_elements_perrun = design_output.outputs.design_elements_perrun
    glm_run_output = glm_run.run()
    print()
