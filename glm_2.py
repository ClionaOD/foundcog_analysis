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

class GLMAnalysis(GLMPathSetter):
    def __init__(self, base_dir, func_deriv, sub, task, brain_mask='', motion_param_deriv='motion_parameters', fwd_deriv='motion_fwd'):
        super().__init__(base_dir, func_deriv, sub, task, brain_mask, motion_param_deriv, fwd_deriv)
        self._get_fnames()
    
    def __str__(self):
        return str(self.paths)


if __name__ == '__main__':
    in_root_path = '/foundcog/dataset_sharing'
    subject_list = ['2001']
    
    for sub in subject_list:
        for task in ['pictures']:
            paths = GLMAnalysis(
                in_root_path,
                'normalized_to_common_space',
                sub,
                task,
                brain_mask='/foundcog/templates/mask/nihpd_asym_02-05_fcgmask_2mm.nii.gz',
            )
            print(paths)