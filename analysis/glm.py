from os import path
import glob

from os import path
import glob

import os
import pickle
from collections import defaultdict

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
        default_value="derivatives/foundcog_preproc",
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
    tr = traits.Float(mandatory=True, desc="Repetition time (TR) in seconds")
    brain_mask = traits.Str(mandatory=True, desc="Path to the brain mask image")

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
    hold = traits.Dict(desc="Placeholder dict")


class GLMDesign(BaseInterface):
    input_spec = GLMDesignInputSpec
    output_spec = GLMDesignOutputSpec

    def _run_interface(self, runtime):
        self._results = {}

        self._check_path_dict()

        if not (self.inputs.repetition_marking) and not (self.inputs.exemplar_marking):
            print(
                "Neither repetition marking nor exemplar marking is set. No repetitions or exemplars will be marked in the design matrix."
            )

        if (self.inputs.exemplar_marking) and (self.inputs.task != "pictures"):
            raise ValueError(
                "Exemplar marking is only applicable for the 'pictures' task."
            )

        if self.inputs.video_tag_marking:
            if self.inputs.task != "videos":
                raise ValueError(
                    "Video tag marking is only applicable for the 'videos' task."
                )

        ## do stuff

        self._results["hold"] = {}
        return runtime

    def _list_outputs(self):
        return self._results

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

    def _get_video_tag_events(self):
        if not path.exists(self.inputs.video_tag_path):
            raise ValueError(
                f"Video tag file not found: {self.inputs.video_tag_path}. Please provide a valid path."
            )

    def _get_exemplar_events(self):
        pass

    def _get_repetition_events(self):
        pass

    def _build_confound_matrix(self):
        pass

    def _fwd_censoring(self):
        pass


## BROKEN ##
class GLMAnalysis(GLMPathSetter):
    def __init__(
        self,
        base_dir,
        func_deriv,
        sub,
        task,
        brain_mask="",
        motion_param_deriv="motion_parameters",
        fwd_deriv="motion_fwd",
    ):
        super().__init__(
            base_dir, func_deriv, sub, task, brain_mask, motion_param_deriv, fwd_deriv
        )
        self._get_fnames()

    def _get_elan_tags(event_df):
        elan_tags = pd.read_pickle("events_per_movie_longlist_new.pickle")
        chosen_trials = ["faces"]  # ,'body_parts','scene','tools']
        # chosen_movies = ['dog','moana','minions_supermarket','forest','bathsong','new_orleans']
        elan_tags = {k.replace(".mp4", ""): v for k, v in elan_tags.items()}
        elan_tags = {
            k: df[df["trial_type"].isin(chosen_trials)] for k, df in elan_tags.items()
        }

        elan = []
        for trial in event_df["trial_type"]:
            if trial not in elan_tags.keys():
                elan.append(event_df[event_df["trial_type"] == trial])
                continue
            elans = elan_tags[trial]
            video_onset = np.array(event_df[event_df["trial_type"] == trial]["onset"])
            for onset in video_onset:
                adjusted_elans = elans.copy()
                adjusted_elans.loc[:, "onset"] = adjusted_elans.loc[:, "onset"] + onset
                elan.append(adjusted_elans)
        elan_df = pd.concat(elan, ignore_index=True)
        elan_df.drop(columns=["magnitude"], inplace=True)
        elan_df.sort_values(by=["onset"], inplace=True, ignore_index=True)
        return elan_df

    def model_run(
        self,
        recorded_tr=0.610,
        brain_mask="/foundcog/templates/mask/nihpd_asym_02-05_fcgmask_2mm.nii.gz",
        fwd_cutoff=0.5,
        rep_marking=True,
        exemplar_marking=False,
        derivs="normalized_to_common_space",
        vid_tags=False,
    ):

        _rep_tag = "_reps" if rep_marking and self.task == "pictures" else ""
        _eg_tag = "_eg" if exemplar_marking and self.task == "pictures" else ""

        # load events first
        events = [pd.read_csv(ev, sep="\t") for ev in self.paths["events"]]

        outpaths = []
        skipruns = []
        for runidx in range(len(events)):
            # Get the functional image for this run
            run_img = self.paths["func"][runidx]

            # get which ses/run we're in
            runnum = self.paths["run_order"][runidx]
            sesnum = self.paths["ses_order"][runidx]

            # BUILD CONFOUNDS - do this first to check if run passes motion threshold
            condf = pd.read_csv(
                self.paths["motion"][runidx], header=None, sep="  ", engine="python"
            )
            conf_names = [
                "motion_x",
                "motion_y",
                "motion_z",
                "rotation_x",
                "rotation_y",
                "rotation_z",
            ]

            # Construct spike regressors for this participant
            # 1. Read in the FWD file
            fwd = pd.read_csv(self.paths["fwd"][runidx])

            # 2. Get indices where this is over cutoff
            # #   Be careful of whether setting 1st or 2nd scan of difference
            # THIS IS FIRST SCAN
            above_idxs = fwd.index[fwd["FramewiseDisplacement"] > fwd_cutoff].values

            # THIS WOULD BE SECOND SCAN
            # above_idxs = above_idxs + 1

            # 3. Construct matrix with nrows=nscans, ncols=nframes_todrop
            spike_arr = np.zeros((len(fwd) + 1, above_idxs.size))
            spike_arr[above_idxs, np.arange(above_idxs.size)] = 1

            if spike_arr.shape[1] > 0:
                conf_names.extend([f"spike_{i}" for i in range(spike_arr.shape[1])])

            # Add spike regressors along axis 1 of condf array
            confounds = np.concatenate((condf, spike_arr), axis=1)
            confounds = np.nan_to_num(confounds)

            # Check to see if number of spikes is over threshold
            discard_thresh = 0.5
            if len(above_idxs) > len(fwd) * discard_thresh:
                print(
                    f"Too much motion in sub-{self.sub} run {runidx+1} of {len(events)}, skipping"
                )
                skipruns.append({"ses": sesnum, "run": runnum, "reason": "motion"})
                continue

            # tags = pd.read_csv(paths['camera'][runidx], sep='\t')
            # confounds = np.concatenate((confounds, tags), axis=1)
            # conf_names.extend(tags.columns.to_list())

            # initialise the model
            model = FirstLevelModel(t_r=recorded_tr, mask_img=brain_mask)

            run_events = events[runidx]
            # relevant for video trials
            run_events["trial_type"] = run_events["trial_type"].str.replace(".mp4", "")

            if vid_tags:
                if self.task != "videos":
                    raise ValueError("vid_tags=True only works for videos")
                run_events = self._get_elan_tags(run_events)

            # use fixation as baseline if it's there (i.e. for pictures) else use the attention getters (for videos)
            if "fixation" in run_events["trial_type"].to_list():
                run_events = run_events[run_events["trial_type"] != "fixation"]
            else:
                run_events = run_events[run_events["trial_type"] != "attention_getter"]

            # remove underscores for easier string slicing down the line
            run_events["trial_type"] = run_events["trial_type"].str.replace("_", "")

            # mark individual exemplars
            if exemplar_marking and "picture" in self.task:
                run_events["stim_file"] = (
                    run_events["stim_file"].str.replace(".png", "").str.replace("_", "")
                )
                run_events.loc[1:, "trial_type"] = run_events.loc[1:, "stim_file"]

            # mark repetitions
            if rep_marking:
                if exemplar_marking:
                    numexemplar = 1
                else:
                    numexemplar = 3 if "picture" in self.task else 1
                tts = run_events.trial_type.to_list()
                dd = defaultdict(int)
                marked_tts = []
                for trial in tts:
                    dd[trial] += 1
                    if dd[trial] <= numexemplar:
                        marked_tts.append(f"{trial}_rep1")
                    else:
                        marked_tts.append(f"{trial}_rep2")
                run_events["trial_type"] = marked_tts

            # BUILD FRAME TIMES - code snippet taken from nilearn
            n_scans = get_data(run_img).shape[3]
            start_time = model.slice_time_ref * model.t_r
            end_time = (n_scans - 1 + model.slice_time_ref) * model.t_r
            frame_times = np.linspace(start_time, end_time, n_scans)

            # MAKE DESIGN MATRIX
            design = make_first_level_design_matrix(
                frame_times,
                events=run_events,
                hrf_model=model.hrf_model,
                drift_model=model.drift_model,
                high_pass=model.high_pass,
                drift_order=model.drift_order,
                fir_delays=model.fir_delays,
                add_regs=confounds,
                add_reg_names=conf_names,
                min_onset=model.min_onset,
            )

            fit = True  # only purpose of this switch is for testing
            if fit:
                # get the functional image
                run_img_loaded = check_niimg(run_img, ensure_ndim=4)

                # fit the model
                model.fit(run_img_loaded, design_matrices=design)

                # save out the inputs to the model alongside it
                model_save = {
                    "model": model,
                    "input_events": run_events,
                    "funcfile": run_img,
                    "fwdcutoff": fwd_cutoff,
                    "rep_marking": rep_marking,
                    "exemplar_marking": exemplar_marking,
                }

                model_path = os.path.abspath(
                    f"sub-{sub}_ses-{sesnum}_task-{task}_run-{runnum}{_eg_tag}{_rep_tag}_elantags_model.pickle"
                )
                # model_path = os.path.join('/foundcog/foundcog_results/elanmodels',f'sub-{self.sub}_ses-{sesnum}_task-{self.task}_run-{runnum}{_eg_tag}{_rep_tag}_elantags_model.pickle')
                with open(model_path, "wb") as f:
                    pickle.dump(model_save, f)
                outpaths.append(model_path)

        # save out the skipped runs
        skippath = []
        if len(skipruns) > 0:
            skipruns = pd.DataFrame(skipruns)
            skip_path = os.path.abspath(
                f"sub-{sub}_task-{task}{_eg_tag}{_rep_tag}_skippedruns.csv"
            )
            # skip_path = os.path.join('/foundcog/foundcog_results/elanmodels',f'sub-{self.sub}_task-{self.task}{_eg_tag}{_rep_tag}_skippedruns.csv')
            skipruns.to_csv(skip_path, index=False)
            skippath.append(skip_path)

        return outpaths, skippath, rep_marking, exemplar_marking


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

    SUB = "2001"
    TASK = "pictures"

    from nipype import Node

    ## SET PATHS
    glm_paths = Node(GLMPathSetter(), name="glm_path_node")
    glm_paths.inputs.base_dir = "/foundcog/dataset_sharing"
    glm_paths.inputs.func_deriv = "normalized_to_common_space"
    glm_paths.inputs.sub = SUB
    glm_paths.inputs.task = TASK

    path_output = glm_paths.run()
    paths = path_output.outputs.paths

    ## GET DESIGN MATRIX
    glm_design = Node(GLMDesign(), name="glm_design_node")
    glm_design.inputs.sub = SUB
    glm_design.inputs.task = TASK
    glm_design.inputs.tr = 0.610
    glm_design.inputs.brain_mask = (
        "/foundcog/templates/mask/nihpd_asym_02-05_fcgmask_2mm.nii.gz"
    )

    glm_design.inputs.repetition_marking = False
    glm_design.inputs.exemplar_marking = True
    glm_design.inputs.gaze_coding = False
    glm_design.inputs.video_tag_marking = False

    glm_design.inputs.paths = paths

    design_output = glm_design.run()
    print()
