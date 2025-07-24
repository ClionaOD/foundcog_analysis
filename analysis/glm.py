from os import path
import glob

class GLMPathSetter():
    def __init__(
            self,
            base_dir, 
            func_deriv,
            sub, 
            task,
            brain_mask='',
            motion_param_deriv='motion_parameters',
            fwd_deriv='motion_fwd',
            derivative_dir='derivatives/foundcog_preproc'
    ):
        self.base_dir = base_dir
        self.func_deriv = func_deriv
        self.sub = sub
        self.task = task
        self.brain_mask = brain_mask
        self.motion_param_deriv = motion_param_deriv
        self.fwd_deriv = fwd_deriv
        self.derivative_dir = derivative_dir

        # Set paths for the experiment
        self._get_fnames()
    
    def _set_expt_paths(self):
        # Set paths for the experiment
        self.func_file = path.join(
            self.base_dir,
            self.derivative_dir,
            self.func_deriv,
            'sub-{sub}',
            'ses-?_run-00?_task-{task}',
            # TODO: fix for other filenames
            'sub-{sub}_ses-?_task-{task}_dir-AP_run-00?_bold_mcf_corrected_flirt.nii.gz'
        )

        self.event_file = path.join(
            self.base_dir, 
            'sub-{sub}',
            'ses-{sesnum}',
            'func',
            'sub-{sub}_ses-{sesnum}_task-{task}_dir-AP_run-{runnum}_events.tsv'
        )

        self.motion_param_file = path.join(
            self.base_dir, 
            self.derivative_dir,
            self.motion_param_deriv,
            'sub-{sub}', 
            'ses-{sesnum}_run-{runnum}_task-{task}',
            'sub-{sub}_ses-{sesnum}_task-{task}_dir-AP_run-{runnum}_bold_mcf.nii.par'
        )

        self.fwd_file = path.join(
            self.base_dir   , 
            self.derivative_dir,
            self.fwd_deriv,
            'sub-{sub}', 
            'ses-{sesnum}_run-{runnum}_task-{task}',
            'fd_power_2012.txt'
        )
    
    def _match_func_to_datafiles(self, funcrunpath):
        info= funcrunpath.split('/')[-1].split('_')
        runnum = [i for i in info if 'run' in i][0].split('-')[-1]
        sesnum = [i for i in info if 'ses' in i][0].split('-')[-1]

        event_file = self.event_file.format(
            sub=self.sub, 
            sesnum=sesnum, 
            task=self.task, 
            runnum=runnum
        )
        assert path.exists(event_file), f"Event file not found: {event_file}"
        motion_param_file = self.motion_param_file.format(
            sub=self.sub, 
            sesnum=sesnum, 
            task=self.task, 
            runnum=runnum
        )
        assert path.exists(motion_param_file), f"Motion parameter file not found: {motion_param_file}"
        fwd_file = self.fwd_file.format(
            sub=self.sub, 
            sesnum=sesnum, 
            task=self.task, 
            runnum=runnum
        )
        assert path.exists(fwd_file), f"FWD file not found: {fwd_file}"
        return {'func': [funcrunpath], 
                'events': [event_file], 
                'motion': [motion_param_file], 
                'fwd': [fwd_file],
                'run_order': [runnum],
                'ses_order': [sesnum],
            }

    def _get_fnames(self):
        self._set_expt_paths()
        funcpaths = glob.glob(self.func_file.format(
            sub=self.sub, 
            task=self.task
        ))

        if len(funcpaths) == 0:
            # try again without the corrected suffix, maybe there were no fmaps
            self.func_file = self.func_file.replace('_corrected_flirt.nii.gz', '_flirt.nii.gz')
            funcpaths = glob.glob(self.func_file.format(
                sub=self.sub, 
                task=self.task
            ))
        
        paths = {}
        for funcrunpath in funcpaths:
            expt_dict_fnames = self._match_func_to_datafiles(funcrunpath)
            for key, value in expt_dict_fnames.items():
                if key in paths:
                    paths[key].extend(value)
                else:
                    paths[key] = value.copy()  # Use copy() to avoid aliasing

        self.paths = paths

def model_run(sub, task, recorded_tr=0.610, brain_mask='/foundcog/templates/mask/nihpd_asym_02-05_fcgmask_2mm.nii.gz', fwd_cutoff=0.5, rep_marking=True, exemplar_marking=False, derivs='normalized_to_common_space', vid_tags=False):    
    
    ### FUNCTIONS DEFINED WITHIN MODULE BECAUSE OF NIPYPE ##

    def _get_elan_tags(event_df):
        elan_tags = pd.read_pickle('events_per_movie_longlist_new.pickle')
        chosen_trials = ['faces']#,'body_parts','scene','tools']
        # chosen_movies = ['dog','moana','minions_supermarket','forest','bathsong','new_orleans']
        elan_tags = {k.replace('.mp4',''):v for k,v in elan_tags.items()}
        elan_tags = {k:df[df['trial_type'].isin(chosen_trials)] for k,df in elan_tags.items()}

        elan = []
        for trial in event_df['trial_type']:
            if trial not in elan_tags.keys():
                elan.append(event_df[event_df['trial_type']==trial])
                continue
            elans = elan_tags[trial]
            video_onset = np.array(event_df[event_df['trial_type']==trial]['onset'])
            for onset in video_onset:
                adjusted_elans = elans.copy()
                adjusted_elans.loc[:,'onset'] = adjusted_elans.loc[:,'onset'] + onset
                elan.append(adjusted_elans)
        elan_df = pd.concat(elan, ignore_index=True)
        elan_df.drop(columns=['magnitude'], inplace=True)
        elan_df.sort_values(by=['onset'], inplace=True, ignore_index=True)
        return elan_df

    import os
    import pickle
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    from nilearn._utils.niimg_conversions import check_niimg
    from nilearn.glm.first_level import (FirstLevelModel,
                                         make_first_level_design_matrix)
    from nilearn.image import get_data

    paths = GLMPathSetter(
        base_dir='/foundcog/dataset_sharing',
        func_deriv=derivs,
        sub=sub,
        task=task,
        brain_mask=brain_mask,
        motion_param_deriv='motion_parameters',
        fwd_deriv='motion_fwd'
    ).paths

    _rep_tag = '_reps' if rep_marking and task=='pictures' else ''
    _eg_tag = '_eg' if exemplar_marking and task=='pictures' else ''
    
    # load events first
    events = [pd.read_csv(ev,sep='\t') for ev in paths['events']]

    outpaths = []
    skipruns = []
    for runidx in range(len(events)):
        # Get the functional image for this run
        run_img = paths['func'][runidx]
        
        # get which ses/run we're in
        runnum = paths['run_order'][runidx]
        sesnum = paths['ses_order'][runidx]

        # BUILD CONFOUNDS - do this first to check if run passes motion threshold
        condf = pd.read_csv(paths['motion'][runidx], header=None, sep='  ', engine='python')
        conf_names = ['motion_x', 'motion_y', 'motion_z', 'rotation_x', 'rotation_y', 'rotation_z']
        
        # Construct spike regressors for this participant
        # 1. Read in the FWD file
        fwd = pd.read_csv(paths['fwd'][runidx])

        # 2. Get indices where this is over cutoff
        # #   Be careful of whether setting 1st or 2nd scan of difference
        # THIS IS FIRST SCAN
        above_idxs = fwd.index[fwd['FramewiseDisplacement']>fwd_cutoff].values
        
        # THIS WOULD BE SECOND SCAN
        # above_idxs = above_idxs + 1

        # 3. Construct matrix with nrows=nscans, ncols=nframes_todrop
        spike_arr = np.zeros((len(fwd)+1,above_idxs.size))
        spike_arr[above_idxs,np.arange(above_idxs.size)] = 1

        if spike_arr.shape[1] > 0:
            conf_names.extend([f'spike_{i}' for i in range(spike_arr.shape[1])])

        # Add spike regressors along axis 1 of condf array
        confounds = np.concatenate((condf,spike_arr),axis=1)
        confounds = np.nan_to_num(confounds)

        # Check to see if number of spikes is over threshold
        discard_thresh = 0.5
        if len(above_idxs) > len(fwd) * discard_thresh:
            print(f'Too much motion in sub-{sub} run {runidx+1} of {len(events)}, skipping')
            skipruns.append({'ses':sesnum, 'run':runnum, 'reason':'motion'})
            continue

        # tags = pd.read_csv(paths['camera'][runidx], sep='\t')
        # confounds = np.concatenate((confounds, tags), axis=1)
        # conf_names.extend(tags.columns.to_list())

        # initialise the model
        model = FirstLevelModel(t_r=recorded_tr, mask_img=brain_mask)

        run_events = events[runidx]
        # relevant for video trials
        run_events['trial_type'] = run_events['trial_type'].str.replace('.mp4','')

        if vid_tags:
            if task != 'videos':
                raise ValueError('vid_tags=True only works for videos')
            run_events = _get_elan_tags(run_events)
        
        # use fixation as baseline if it's there (i.e. for pictures) else use the attention getters (for videos)
        if 'fixation' in run_events['trial_type'].to_list():
            run_events = run_events[run_events['trial_type']!='fixation']
        else:
            run_events = run_events[run_events['trial_type']!='attention_getter']
        
        # remove underscores for easier string slicing down the line
        run_events['trial_type'] = run_events['trial_type'].str.replace('_','')
        
        # mark individual exemplars
        if exemplar_marking and 'picture' in task:
            run_events['stim_file'] = run_events['stim_file'].str.replace('.png','').str.replace('_','')
            run_events.loc[1:,'trial_type'] = run_events.loc[1:,'stim_file']

        # mark repetitions
        if rep_marking:    
            if exemplar_marking:
                numexemplar = 1
            else:
                numexemplar = 3 if 'picture' in task else 1
            tts=run_events.trial_type.to_list()
            dd = defaultdict(int)
            marked_tts = []
            for trial in tts:
                dd[trial] += 1
                if dd[trial] <= numexemplar:
                    marked_tts.append(f'{trial}_rep1')
                else:
                    marked_tts.append(f'{trial}_rep2')
            run_events['trial_type'] = marked_tts

        # BUILD FRAME TIMES - code snippet taken from nilearn
        n_scans = get_data(run_img).shape[3]
        start_time = model.slice_time_ref * model.t_r
        end_time = (n_scans - 1 + model.slice_time_ref) * model.t_r
        frame_times = np.linspace(start_time, end_time, n_scans)

        # MAKE DESIGN MATRIX
        design = make_first_level_design_matrix(    frame_times,
                                                    events=run_events,
                                                    hrf_model=model.hrf_model,
                                                    drift_model=model.drift_model,
                                                    high_pass=model.high_pass,
                                                    drift_order=model.drift_order,
                                                    fir_delays=model.fir_delays,
                                                    add_regs=confounds,
                                                    add_reg_names=conf_names,
                                                    min_onset=model.min_onset
                                                    )
        
        fit=True # only purpose of this switch is for testing
        if fit:
            # get the functional image
            run_img_loaded = check_niimg(run_img, ensure_ndim=4)
            
            # fit the model
            model.fit(run_img_loaded, design_matrices=design)

            # save out the inputs to the model alongside it
            model_save = {
                'model':model,
                'input_events':run_events,
                'funcfile':run_img,
                'fwdcutoff':fwd_cutoff,
                'rep_marking':rep_marking,
                'exemplar_marking':exemplar_marking
            }
            
            # model_path = os.path.abspath(f'sub-{sub}_ses-{sesnum}_task-{task}_run-{runnum}{_eg_tag}{_rep_tag}_elantags_model.pickle')
            model_path = os.path.join('/foundcog/foundcog_results/elanmodels',f'sub-{sub}_ses-{sesnum}_task-{task}_run-{runnum}{_eg_tag}{_rep_tag}_elantags_model.pickle')
            with open(model_path,'wb') as f:
                pickle.dump(model_save, f)
            outpaths.append(model_path)

    # save out the skipped runs
    skippath = []
    if len(skipruns) > 0:
        skipruns = pd.DataFrame(skipruns)
        # skip_path = os.path.abspath(f'sub-{sub}_task-{task}{_eg_tag}{_rep_tag}_skippedruns.csv')
        skip_path = os.path.join('/foundcog/foundcog_results/elanmodels',f'sub-{sub}_task-{task}{_eg_tag}{_rep_tag}_skippedruns.csv')
        skipruns.to_csv(skip_path,index=False)
        skippath.append(skip_path)

    return outpaths, skippath, rep_marking, exemplar_marking

def single_sub_betas(sub,task,subrunpaths, rep_marking, exemplar_marking):
    
    def _get_betas(model,colind, cov=None):
        """
        Our usual beta extracting code
        """
        import numpy as np

        labels=model.labels_[0]
        regression_result=model.results_[0]
        effect = np.zeros((labels.size))
        vcov = np.zeros((labels.size))

        for label_ in regression_result.keys():
            if cov is None:
                cov = regression_result[label_].cov
                
            label_mask = labels == label_
            if label_mask.any():
                resl = regression_result[label_].theta[colind]
                vcov[label_mask] = regression_result[label_].vcov(column=colind)
                effect[label_mask]=resl


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
            model=model['model']
        numvert = len(model.labels_[0])
        
        # get columns of interest in design
        cols=model.design_matrices_[0].columns
        
        # get the conditions in this run - must be in our ideal list
        conditions = []
        for c in cols:
            if (c.split('_')[0] in all_conds) or (c[:-1] in all_conds) or (c.split('_')[0][:-1] in all_conds):
                conditions.append(c)
        
        numcond = len(conditions)

        # Putting this straight into two numpy arrays so we don't need to convert later
        vol_betas=np.zeros((numvert, numcond))
        vol_vcov=np.zeros((numvert, numcond))
        cov = None

        beta_labels = []
        for ind, trial_type in enumerate(conditions):
            colind=cols.get_loc(trial_type)
            beta_labels.append(trial_type)
            effect, vcov, cov = _get_betas(model,colind,cov=cov)
            vol_betas[:,ind]=effect
            vol_vcov[:,ind]=vcov

        # Diagnostics - dump cov to figure

        con = np.zeros((1,cov.shape[0]))
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
        cov = cov[list(range(numcond+1))+[-1],:]
        cov = cov[:,list(range(numcond+1))+[-1]]
        im=ax[0].imshow(cov)
        plt.colorbar(im)

        x=np.arange(-2,2,0.05)    
        a=np.zeros((numcond+1,len(x)))            
        for colind in range(numcond+1):
            con = np.zeros((1,numcond+2))
            con[0,colind]=1
            for qind,q in enumerate(x):
                con[0,-1] = q
                a[colind,qind]=np.dot(con, np.dot(cov,np.transpose(con)))
        ax[1].plot(x,a.T)

        # plt.savefig(f'sub-{sub}_task-{task}_covariance{_eg_tag}{_rep_tag}.png')
        plt.close()


        return (beta_labels,vol_betas, vol_vcov, cov)
    
    import os

    if len(subrunpaths) == 0:
        return []

    _rep_tag = '_reps' if rep_marking else ''
    _eg_tag = '_eg' if exemplar_marking else ''

    outpaths = []

    beta_path = os.path.abspath(f'sub-{sub}_task-{task}_voxelwisebetas_withvcov{_eg_tag}{_rep_tag}.pickle')
    
    # used to compare to ideal run
    if task == 'videos':
        allconds=['bathsong', 'dog', 'neworleans', 'minionssupermarket', 'forest', 'moana']
    elif task == 'pictures':
        allconds=[ 'seabird', 'crab',
                        'dishware', 'food',
                        'tree', 'squirrel',
                        'rubberduck', 'towel',
                        'shelves', 'shoppingcart',
                        'cat','fence'
                    ]

    # get each run's beta values and conditions modelled for this run
    beta_save = {}
    for run in subrunpaths:
        _colnames, _runbetas, _runvcov, _runcov = _mvpa_betas(run,allconds)
        beta_save[os.path.basename(run)] ={'betas':_runbetas, 'vcov':_runvcov, 'colnames':_colnames, 'cov': _runcov}
    
    if len(beta_save) >0:
        with open(beta_path,'wb') as f:
            # pickle.dump(beta_save, f)
            pass
        outpaths.append(beta_path)
    
    return outpaths

if __name__=='__main__':
    import glob
    
    in_root_path = '/foundcog/dataset_sharing'
    subject_list = ['2001']
    
    for sub in subject_list:
        for task in ['pictures']:
            paths=model_run(
                sub,
                task,
                fwd_cutoff=1.5,
                recorded_tr=0.610,
                brain_mask='/foundcog/templates/mask/nihpd_asym_02-05_fcgmask_2mm.nii.gz',
                derivs='normalized_to_common_space',
                rep_marking=True,
                exemplar_marking=False)
            print(paths)

    rep_marking = False
    exemplar_marking = True
    for sub in subject_list:
        for task in ['pictures']:#,'videos']:
            _rep_tag = '_reps' if rep_marking else ''
            _eg_tag = '_eg' if exemplar_marking else ''

            subrunpaths = glob.glob(f'/foundcog/foundcog_results/{task}/models/sub-{sub}_task-{task}/*{_eg_tag}{_rep_tag}_model.pickle')
            outpaths=single_sub_betas(sub,task,subrunpaths,rep_marking=rep_marking,exemplar_marking=exemplar_marking)
            print(outpaths)