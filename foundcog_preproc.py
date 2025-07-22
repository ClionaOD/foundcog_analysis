
import glob
import json
import os
from datetime import datetime
from os import path

import git
import nibabel as nib
import numpy as np
from bids.layout import BIDSLayout
from nipype import Function, JoinNode, MapNode, Node, Workflow, config
from nipype.interfaces import ants, fsl
from nipype.interfaces.io import DataSink, SelectFiles
from nipype.interfaces.utility import IdentityInterface
from niworkflows.interfaces.images import SignalExtraction

from preproc.bold_preproc import get_wf_bold_preproc

SINGLE_THREADED = False

config.enable_debug_mode()

exclude_subjects=  []

# Exclude series with fewer volumes than this
min_fmri_volumes=50

# Default memory per job
DEFAULT_MEM = '100'

# Change this to the path where you have the BIDS dataset
experiment_dir = path.abspath(path.join('/foundcog','dataset_sharing'))
database_path=path.abspath(path.join(experiment_dir,'bidsdatabase'))
layout = BIDSLayout(experiment_dir, database_path=database_path)

# Path to the reference files, such as templates
reference_files_path = path.abspath(path.join('.','preproc','reference_files'))

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

# # TR of functional images
with open(path.join(experiment_dir,'task-pictures_bold.json'), 'rt') as fp:
    task_info = json.load(fp)
TR = task_info['RepetitionTime']


# subject_list=subject_list[1:] # for testing, just one subject

print(f'Subjects {subject_list}')
print(f'Sessions {session_list}')
print(f'Tasks {task_list}')
print(f'TR is {TR}')


# Open json file listing manual selection of flirt dof
# add any additional subject's ID automatically
# save out the new version
# This is a manual selection of the dof for flirt normalisation, most subjects will have 12, but some may have 6 or 9.
with open(os.path.abspath(path.join(reference_files_path,'flirt_dof_manualselection.json')), 'rt') as f:
    flirt_dof_bysub = json.load(f)

for sub in subject_list:
    if sub not in list(flirt_dof_bysub.keys()):
        flirt_dof_bysub[sub] = []

with open(os.path.abspath(path.join(reference_files_path,'flirt_dof_manualselection.json')), 'w') as f:
    json.dump(flirt_dof_bysub, f)

# Make a list of functional steps to do
iter_items= {}

# Templates
func_file = path.join('sub-{subject_id}', 'ses-{session}', 'func', 'sub-{subject_id}_ses-{session}_task-{task_name}_dir-AP_run-{run}_bold.nii.gz')
fmap_file = path.join('sub-{subject_id}', 'ses-{session}', 'fmap', 'sub-{subject_id}_ses-{session}_dir-{dir}_run-???_epi.nii.gz')
event_file = path.join('sub-{subject_id}', 'ses-{session}', 'func', 'sub-{subject_id}_ses-{session}_task-{task_name}_dir-AP_run-{run}_events.tsv')

for sub in subject_list:

    # Check for fieldmap
    for dir in ['AP', 'PA']:
        info = {'subject_id':sub, 'session':'?', 'dir':dir}
        fmap_files = glob.glob(path.join(experiment_dir,fmap_file.format(**info)))

        if not fmap_files:
            print(f'Warning no fieldmaps found for {sub} phase-encoding direction {dir}')
            
    iter_items[sub]= {'ses':[], 'task': [], 'run':[]}
    for ses in session_list:
        for task in task_list:
            for run in run_list:
                info = {'subject_id':sub, 'session':ses, 'task_name':task, 'run': run}
                target_file = func_file.format(**info)
                if path.isfile(path.join(experiment_dir, target_file)):
                    im=nib.load(path.join(experiment_dir, target_file))
                    if im.shape[3]>min_fmri_volumes:
                        iter_items[sub]['ses'].append(str(ses))
                        iter_items[sub]['task'].append(str(task))
                        iter_items[sub]['run'].append(str(run))

# Remove exclude_subjects
for sub in exclude_subjects:
    if sub in iter_items:
        iter_items.pop(sub)

# Setup
omp_nthreads = 8
mem_gb = {"filesize": 1, "resampled": 1, "largemem": 1}

# save which git hash we ran this time for version control
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
current_time = datetime.now()
timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
f = open(f'.githash/{timestamp}_githash.txt','w')
f.write(sha)
f.close()

# MAIN WORKFLOW
#  different sessions/runs/tasks per subject
subs_run = []
for sub, sub_items in iter_items.items():

    # absolute path from experiment_dir will be used in line516 below
    working_dir = f'workingdir/{sub}'
    output_dir = f'derivatives/foundcog_preproc'

    # INPUT DATA
    infosource_sub = Node(IdentityInterface(fields=['subject_id']), name="infosource_sub")
    infosource_sub.iterables = [('subject_id', [sub])]    

    infosource = Node(IdentityInterface(fields=['session', 'task_name', 'run']),
                    name="infosource")
    infosource.iterables = [('session', sub_items['ses']),('task_name', sub_items['task']),
                            ('run', sub_items['run']),
                            ]

    infosource.synchronize = True # synchronised stepping through each of the iterable lists

    # SelectFiles - to grab the data (alternativ to DataGrabber)
    templates = {'func': func_file,'events':event_file}
    selectfiles = Node(SelectFiles(templates,
                                base_directory=experiment_dir),
                    name="selectfiles")
    
    # Datasink - creates output folder for important outputs
    #  for stuff output once per fMRI run
    datasink_run = Node(DataSink(base_directory=experiment_dir,
                            container=output_dir),
                    name="datasink_run")
    #  for stuff output once per subject
    datasink_subj = Node(DataSink(base_directory=experiment_dir,
                            container=output_dir),
                    name="datasink_subj")
    #  for stuff output once per dof
    datasink_dof = Node(DataSink(base_directory=experiment_dir,
                            container=output_dir),
                    name="datasink_dof")

    # Distortion correction using topup
    join_fmap = JoinNode(IdentityInterface(fields = ['subject']), name='join_fmap', joinsource='infosource', joinfield='subject', unique=True)

    # Get sessions with fmap, in order for selection by session later
    fmap_files={}
    for sess_tmp in session_list:
        sess_fmaps = layout.get(subject=sub,session=sess_tmp, datatype='fmap', extension=['.nii', '.nii.gz'] )
        if len(sess_fmaps) > 1:
            if not sess_tmp in fmap_files:
                fmap_files[sess_tmp] = []
            fmap_files[sess_tmp].extend(sess_fmaps)

    if not fmap_files:
        are_fmaps=False
    else:
        are_fmaps=True

    # Get fmap for this session
    def bg_fmap_files(experiment_dir, database_path, subject_id, fmap_session):
        from os import path
        from bids.layout import BIDSLayout

        layout = BIDSLayout(experiment_dir, database_path=database_path)

        return layout.get(subject=subject_id, session=fmap_session, datatype='fmap', extension=['.nii', '.nii.gz'] )

    # Pick the output from the topup run corresponding to the right session
    def select_by_session(session, applytopup_method, out_fieldcoef, out_movpar, out_enc_file, out_corrected):
        sessind = int(session)-1

        if len(applytopup_method)<2:
            sessind=0             # if we only have one fieldmap, use it

        return applytopup_method[sessind], out_fieldcoef[sessind], out_movpar[sessind],out_enc_file[sessind], out_corrected[sessind]

    def postjoindump(joinedthing):
        print(f'Joined thing is {joinedthing}')

    if are_fmaps:
        # Process fmap for every session from this subject
        bg_fmap = Node(IdentityInterface(fields=['fmap_session']), name='bg_fmap') 
        bg_fmap.iterables = [('fmap_session', list(fmap_files.keys()))]

        bg_fmap_files_node = Node(Function(function=bg_fmap_files, input_names=['experiment_dir','database_path','subject_id','fmap_session'], output_names=['fmap']), name='bg_fmap_files')
        bg_fmap_files_node.inputs.experiment_dir = experiment_dir
        bg_fmap_files_node.inputs.database_path = database_path

        hmc_fmaps = MapNode(fsl.MCFLIRT(), iterfield='in_file', name='hmc_fmaps')
        mean_fmaps = MapNode(fsl.maths.MeanImage(), iterfield='in_file', name='mean_fmaps')
        merge_fmaps = Node(fsl.Merge(dimension='t'), name='merge_fmaps')


    # Gather all of bolds at end
    joinbold=JoinNode(IdentityInterface(fields=['in_file']), joinsource='infosource', joinfield='in_file', name='joinnode')
    
    runmean = Node(fsl.maths.MeanImage(), name='runmean')
    runstd = Node(fsl.maths.StdImage(), name='runstd')
    snr_func = Node(fsl.maths.BinaryMaths(operation='div'), name='snr_func')
    
    def get_coreg_reference(in_files):
        print(f'get_coreg_reference received {in_files}')
        # Pick reference for coreg - order by pref rest10, rest5, videos, pictures
        # Resting state was an asleep scan, so this will be used preferentially
        for pref in ['rest10', 'rest5', 'videos', 'pictures']:
            res = [out_file for out_file in in_files if pref in out_file]
            if (res):
                return res[0]

        # If none of the above
        return in_files[0]

    def select_fmaps(in_files):
        import json
        import os

        import nibabel as nib
        import numpy as np

        remap={'i':'x', 'i-':'x-', 'j':'y', 'j-':'y-', 'k':'z', 'k-':'z-'}
        # Need affines
        ahs={}
        encoding_direction={}
        readout_times={}
        for fmap in in_files:
            affine = np.round(nib.load(fmap).affine,decimals=3)
            print(fmap)
            print(affine)
            affine.flags.writeable = False
            ah = hash(affine.data.tobytes()) # hash of affine matrix used as key
            if not ah in ahs:
                ahs[ah]=[]
                readout_times[ah]=[]
                encoding_direction[ah]=[]
            ahs[ah].append(fmap)


            fmap_s = os.path.splitext(fmap)
            if fmap_s[1]=='.gz':
                fmap_s = os.path.splitext(fmap_s[0])
            with open( fmap_s[0] + '.json', 'r') as f:
                fmap_json = affine = json.load(f)
                readout_times[ah].append(fmap_json['EffectiveEchoSpacing'])
                encoding_direction[ah].append(remap[fmap_json['PhaseEncodingDirection']])
        
        longest_key = max(ahs, key= lambda x: len(set(ahs[x])))

        print(ahs)
        # TODO: Need to adjust so that session-specific fieldmaps are used
        if len(ahs[longest_key])<2:
            ahs={'all': [x for k,v in ahs.items() for x in v ]} 
            encoding_direction={'all': [x for k,v in encoding_direction.items() for x in v ]} 
            readout_times={'all': [x for k,v in readout_times.items()  for x in v]} 
            longest_key='all'

        applytopup_method='jac'

        print(f'Affines {ahs} longest is {longest_key} encoding directions {encoding_direction[longest_key]} readout times {readout_times[longest_key]}')
        return ahs[longest_key], encoding_direction[longest_key], readout_times[longest_key], applytopup_method


    select_fmaps_node = Node(Function(function=select_fmaps, input_names=['in_files'], output_names=['out_files', 'encoding_direction', 'readout_times', 'applytopup_method']), name='select_fmaps')
    topup = Node(fsl.TOPUP(), name='topup')
    select_by_session_node = JoinNode(
                                Function(function=select_by_session, 
                                    input_names=["session", 'applytopup_method', "out_fieldcoef", "out_movpar", "out_enc_file", "out_corrected"], 
                                    output_names=['applytopup_method', "out_fieldcoef", "out_movpar", "out_enc_file", "out_corrected"]),
                                joinsource='bg_fmap', joinfield=['applytopup_method', "out_fieldcoef", "out_movpar", "out_enc_file", "out_corrected"],
                                name='select_by_session_node')

    applytopup = Node(fsl.ApplyTOPUP(), name='applytopup')

    get_coreg_reference_node = Node(Function(function=get_coreg_reference, input_names=['in_files'], output_names=['reference']), name='get_coreg_reference')

    coreg_runs = Node(fsl.FLIRT(), name='coreg_runs')

    coreg_runs_6dof = Node(fsl.FLIRT(dof=6), name='coreg_runs_6dof')

    coreg_singleband_to_submean = Node(fsl.FLIRT(dof=6), name='coreg_singleband_to_submean')

    apply_xfm = Node(fsl.preprocess.ApplyXFM(), name='apply_coreg_runs')
    apply_xfm_to_mean= Node(fsl.preprocess.ApplyXFM(), name='apply_coreg_runs_to_mean')
    apply_xfm_to_snr = Node(fsl.preprocess.ApplyXFM(), name='apply_coreg_snrs')

    submean = JoinNode(ants.AverageImages(dimension=3, normalize=False),  joinsource = 'infosource', joinfield='images', name='submean')
    
    # Location of template file
    ## TODO: paths
    if sub[0] == '9':
        template = '/foundcog/templates/mask/nihpd_asym_08-11_t2w_fcgmasked.nii.gz'
        template_for_norm = '/foundcog/templates/nihpd_asym/nihpd_asym_08-11_t2w.nii'
        template_for_norm_t1w = '/foundcog/templates/nihpd_asym/nihpd_asym_08-11_t1w.nii'
        template_for_norm_2mm = '/foundcog/templates/mask/nihpd_asym_08-11_t2w_2mm.nii.gz'
    else:
        template = '/foundcog/templates/mask/nihpd_asym_02-05_t2w_fcgmasked.nii.gz'
        template_for_norm = '/foundcog/templates/nihpd_asym/nihpd_asym_02-05_t2w.nii'
        template_for_norm_t1w = '/foundcog/templates/nihpd_asym/nihpd_asym_02-05_t1w.nii'
        template_for_norm_2mm = '/foundcog/templates/mask/nihpd_asym_02-05_t2w_2mm.nii.gz'

    # Location of roi file
    schaefer_roi_labels = np.arange(1,401).tolist()

    # TODO: change this to Julich and Schaefer paths
    if sub[0] == '9':
        schaefer_roi = '/foundcog/templates/rois/Schaefer2018_400Parcels_7Networks_order_space-nihpd-08-11_2mm.nii.gz'
        a424_roi = '/foundcog/templates/rois/A424_space-nihpd-08-11_2mm.nii.gz'
    else:
        schaefer_roi = '/foundcog/templates/rois/Schaefer2018_400Parcels_7Networks_order_space-nihpd-02-05_2mm.nii.gz'
        a424_roi = '/foundcog/templates/rois/A424_space-nihpd-02-05_2mm.nii.gz'
 
    # get labels for 424 - derive from file to make sure they correspond
    a424_roi_labels = np.unique(np.asanyarray(nib.load(a424_roi).dataobj))
    a424_roi_labels = list(a424_roi_labels[a424_roi_labels != 0])

    def flirt_fig(in_file,template):
        import os

        from matplotlib import pyplot as plt
        from nilearn import plotting

        plotting.plot_anat(in_file, draw_cross=False).add_edges(template)
        out_file = os.path.abspath("flirt_reg.png")
        plt.savefig(out_file)
        return out_file

    # Normalisation
    input_matrix_file = path.join(reference_files_path, 'flirt_in_matrix.mat')
    flirt_dof_list=[6,9,12]  # rigid, scaling, affine
    flirt_to_template = Node(fsl.FLIRT(bins=640, cost_func='mutualinfo', reference=template, in_matrix_file=input_matrix_file),
        iterables=("dof", flirt_dof_list),
        name=f'flirt')
    #  QA figure
    flirt_fig_node = Node(Function(function=flirt_fig, input_names=['in_file', 'template'], output_names='out_file'), name=f'flirt_fig')
    flirt_fig_node.inputs.template = template
    
    # Normalisation from single band
    if are_fmaps:
        flirt_to_template_singleband = Node(fsl.FLIRT(bins=640, cost_func='mutualinfo', reference=template, in_matrix_file=input_matrix_file),
            iterables=("dof", flirt_dof_list),
            name=f'flirt_singleband')


    # Normalise to common space and smooth
    if flirt_dof_bysub[sub]:
        flirt_to_template_manualselection = Node(fsl.FLIRT(bins=640, dof=flirt_dof_bysub[sub][0], cost_func='mutualinfo', reference=template, in_matrix_file=input_matrix_file),
            name=f'flirt_manualselection')
        flirt_fig_node_manualselection = Node(Function(function=flirt_fig, input_names=['in_file', 'template'], output_names='out_file'), name=f'flirt_fig_manualselection')
        flirt_fig_node_manualselection.inputs.template = template
        
        combine_xfms_manual_selection=  Node(fsl.utils.ConvertXFM(concat_xfm = True), name=f'combine_xfms_manual_selection')
        
        normalize_epi = Node(fsl.preprocess.ApplyXFM(reference=template_for_norm_2mm, apply_xfm=True), name=f'normalization_manualselection')
        normalize_epi.plugin_args =  {'overwrite':True, 'sbatch_args': '--mem=15000'}
        
        smooth_epi = Node(fsl.Smooth(fwhm=8.0), name="smoothing") 
        smooth_epi.plugin_args =  {'overwrite':True, 'sbatch_args': '--mem=8000'}

        # Normalise tSNR - performed with dof manually chosen for epi normalisation
        normalize_snr = Node(fsl.preprocess.ApplyXFM(reference=template_for_norm_2mm, apply_xfm=True), name=f'normalize_snr')

    # Niworkflows SignalExtraction for ROI Extraction
    roi_extraction = Node(SignalExtraction(label_files=schaefer_roi, class_labels=schaefer_roi_labels), name=f'roi_extraction') 
    roi_extraction.plugin_args =  {'overwrite':True, 'sbatch_args': '--mem=16G'}

    roi_extraction_a424 = Node(SignalExtraction(label_files=a424_roi, class_labels=a424_roi_labels), name=f'roi_extraction_a424') 
    roi_extraction_a424.plugin_args =  {'overwrite':True, 'sbatch_args': '--mem=16G'}

    def get_globals(in_file):
        '''Calculate global signal timecourse with mask from nilearn's compute_epi_mask'''
        import os

        import nibabel
        import pandas as pd
        from nilearn import masking

        im = nibabel.load(in_file)
        mask_img = masking.compute_epi_mask(im)
        globals = masking.apply_mask(im, mask_img).mean(axis=1)
        out_file = os.path.abspath('compartment_timecourses.tsv')
        df = pd.DataFrame({'globals': globals})
        df.to_csv(out_file, sep='\t', index=False)
        return out_file
    
    # Calculate globals
    globals_node = Node(Function(function=get_globals, input_names='in_file', output_names='out_file'), name=f'out_file')

    # BOLD preprocessing workflow
    bold_preproc = get_wf_bold_preproc(experiment_dir, working_dir, output_dir)

    # Base workflow
    preproc = Workflow(name='preproc')
    preproc.base_dir = path.join(experiment_dir, working_dir, output_dir)

    preproc.connect([(infosource_sub, infosource, [('subject_id', 'subject_id')]),])

    preproc.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                    ('session', 'session'),
                                    ('task_name', 'task_name'),
                                    ('run', 'run')])
                                    ])
    preproc.connect([(infosource, join_fmap, [('subject_id', 'subject')])])

    preproc.connect([(selectfiles, bold_preproc, [('func','inputnode.func')])])

    # Calculate PE polar distortion correction field using topup
    if are_fmaps:
        preproc.connect([(infosource_sub, bg_fmap_files_node, [('subject_id', 'subject_id')])])
        preproc.connect([(bg_fmap, bg_fmap_files_node, [('fmap_session', 'fmap_session')])])
        preproc.connect([(bg_fmap_files_node, select_fmaps_node, [('fmap', 'in_files')])])
        preproc.connect([(select_fmaps_node, hmc_fmaps, [('out_files', 'in_file')])])
        preproc.connect([(hmc_fmaps, mean_fmaps, [('out_file', 'in_file')])])
        preproc.connect([(mean_fmaps, merge_fmaps, [('out_file', 'in_files')])])
        preproc.connect([(select_fmaps_node, topup, [('readout_times', 'readout_times')])])
        preproc.connect([(select_fmaps_node, topup, [('encoding_direction', 'encoding_direction')])])
        preproc.connect([(merge_fmaps, topup, [('merged_file', 'in_file')])])
    
        # Apply topup
        preproc.connect(bold_preproc, "outputnode.func", applytopup, "in_files")
    
        # Join the fieldmaps processed for the each of the two sessions 
        #  and then pick the fieldmap from the session that matches the current file
        preproc.connect(infosource, "session", select_by_session_node, "session")
        preproc.connect(topup, "out_fieldcoef", select_by_session_node, "out_fieldcoef")
        preproc.connect(topup, "out_movpar", select_by_session_node, "out_movpar")
        preproc.connect(topup, "out_enc_file", select_by_session_node, "out_enc_file")
        preproc.connect(topup, "out_corrected", select_by_session_node, "out_corrected")
        preproc.connect(select_fmaps_node, "applytopup_method", select_by_session_node, "applytopup_method")
    
        # Pass this to applytopup
        preproc.connect(select_by_session_node, "out_fieldcoef", applytopup, "in_topup_fieldcoef")
        preproc.connect(select_by_session_node, "out_movpar", applytopup, "in_topup_movpar")
        preproc.connect(select_by_session_node, "out_enc_file", applytopup, "encoding_file")
        preproc.connect(select_by_session_node, "applytopup_method", applytopup, "method")

        preproc.connect(applytopup, "out_corrected", runmean, "in_file")
        preproc.connect(applytopup, "out_corrected", runstd, "in_file")
    else:
        # No field maps!
        preproc.connect(bold_preproc, "outputnode.func", runmean, "in_file")
        preproc.connect(bold_preproc, "outputnode.func", runstd, "in_file")


    # connect outputs of runmean and runstd to the div operator
    preproc.connect(runmean, "out_file", snr_func, "in_file")
    preproc.connect(runstd, "out_file", snr_func, "operand_file")
    # save to datasink
    preproc.connect(snr_func, "out_file", datasink_run, "snr")

    preproc.connect(runmean, "out_file", joinbold, "in_file")
    preproc.connect(joinbold, "in_file", get_coreg_reference_node, "in_files")
    
    # Default is 12 dof between session means
    preproc.connect(runmean, "out_file", coreg_runs, "in_file")
    preproc.connect(get_coreg_reference_node, "reference", coreg_runs, "reference")

    # Also calculate rigid body coreg between session means, for comparison later (not used in later pipeline)
    preproc.connect(runmean, "out_file", coreg_runs_6dof, "in_file")
    preproc.connect(get_coreg_reference_node, "reference", coreg_runs_6dof, "reference")

    # Reorient runs to space of reference run
    preproc.connect(get_coreg_reference_node, "reference", apply_xfm, "reference")
    
    if are_fmaps:
        preproc.connect(applytopup, "out_corrected", apply_xfm, "in_file")
    else:
        preproc.connect(bold_preproc, "outputnode.func", apply_xfm, "in_file")
        
    preproc.connect(coreg_runs, "out_matrix_file",apply_xfm, "in_matrix_file")

    # Reorient tSNR map to space of reference run
    preproc.connect(get_coreg_reference_node, "reference", apply_xfm_to_snr, "reference")
    preproc.connect(snr_func, "out_file", apply_xfm_to_snr, "in_file")
    preproc.connect(coreg_runs, "out_matrix_file",apply_xfm_to_snr, "in_matrix_file")
    
    # Reorient means to space of reference run
    preproc.connect(get_coreg_reference_node, "reference", apply_xfm_to_mean, "reference")
    preproc.connect(runmean, "out_file", apply_xfm_to_mean, "in_file")
    preproc.connect(coreg_runs, "out_matrix_file",apply_xfm_to_mean, "in_matrix_file")

    # Mean across runs
    preproc.connect(apply_xfm_to_mean, "out_file", submean, "images")

    # Co-register single band to submean, so this can be used for normalisation
    if are_fmaps:
        preproc.connect(select_by_session_node, "out_corrected", coreg_singleband_to_submean, "in_file")
        preproc.connect(submean, "output_average_image", coreg_singleband_to_submean, "reference")

    # Run once per dof (6,9,12)
    preproc.connect(submean, "output_average_image", flirt_to_template, "in_file")
    preproc.connect(flirt_to_template, "out_file", datasink_dof, f"submean_affineflirt")
    preproc.connect(flirt_to_template, "out_matrix_file", datasink_dof, f"submean_affineflirt_matrix")
    preproc.connect(flirt_to_template, "out_file", flirt_fig_node, "in_file")
    preproc.connect(flirt_fig_node, "out_file", datasink_dof, f"submean_affineflirt_figure")

    if are_fmaps:
        # Normalisation using singleband, once per dof (6,9,12)
        preproc.connect(coreg_singleband_to_submean, "out_file", flirt_to_template_singleband, "in_file")

    if flirt_dof_bysub[sub]:
        selected_dof = flirt_dof_bysub[sub][0]
        preproc.connect(submean, "output_average_image", flirt_to_template_manualselection, "in_file")
        preproc.connect(flirt_to_template_manualselection, "out_file", datasink_subj, f"submean_affineflirt_manualselection")
        preproc.connect(flirt_to_template_manualselection, "out_matrix_file", datasink_subj, f"submean_affineflirt_matrix_manualselection")
        preproc.connect(flirt_to_template_manualselection, "out_file", flirt_fig_node_manualselection, "in_file")
        preproc.connect(flirt_fig_node_manualselection, "out_file", datasink_subj, f"submean_affineflirt_figure_manualselection")
        
        # Combine coreg of run to submean and normalisation derived from EPIs into a single transform
        preproc.connect(coreg_runs, "out_matrix_file", combine_xfms_manual_selection, "in_file")
        preproc.connect(flirt_to_template_manualselection, "out_matrix_file",combine_xfms_manual_selection, "in_file2")

        # Apply combined transform to SNR
        preproc.connect(snr_func, "out_file", normalize_snr, "in_file")
        preproc.connect(combine_xfms_manual_selection, "out_file", normalize_snr, "in_matrix_file")
        
        preproc.connect(normalize_snr, "out_file", datasink_run, "snr_normalized_to_common_space") # and send to datasink

        # Apply combined transform to EPIs
        if are_fmaps:
            preproc.connect(applytopup, "out_corrected", normalize_epi, "in_file")
        else:
            preproc.connect(bold_preproc, "outputnode.func", normalize_epi, "in_file")
            
        preproc.connect(combine_xfms_manual_selection, "out_file", normalize_epi, "in_matrix_file")

        preproc.connect(normalize_epi, "out_file", datasink_run, "normalized_to_common_space") # and send to datasink

        # Smoothing
        preproc.connect(normalize_epi, "out_file", smooth_epi, "in_file")
        preproc.connect(smooth_epi, "smoothed_file", datasink_run, "smoothing")

        # ROI Extraction
        preproc.connect(smooth_epi, "smoothed_file", roi_extraction, "in_file")
        preproc.connect(roi_extraction, "out_file", datasink_run, "roi_extraction")

        preproc.connect(smooth_epi, "smoothed_file", roi_extraction_a424, "in_file")
        preproc.connect(roi_extraction_a424, "out_file", datasink_run, "roi_extraction_a424")

        # Calculate globals
        preproc.connect(smooth_epi, "smoothed_file", globals_node, "in_file")
        preproc.connect(globals_node, "out_file", datasink_run, "globals")
     
    else:
        print(f'No dof selected for sub {sub} yet - Please select flirt dof and re-run the pipeline')

    # GRAPHS
    # Create preproc output graph
    preproc.write_graph(graph2use='colored', format='png', simple_form=True)

    # Visualize the graph
    from IPython.display import Image
    Image(filename=path.join(preproc.base_dir, 'preproc', 'graph.png'))
    # Visualize the detailed graph
    preproc.write_graph(graph2use='flat', format='png', simple_form=True)
    Image(filename=path.join(preproc.base_dir, 'preproc', 'graph_detailed.png'))

    # RUN
    if SINGLE_THREADED:
        preproc.run()
    else:
        # Or SLURM?
        preproc.run(plugin='SLURMGraph', plugin_args = {'dont_resubmit_completed_jobs': False, 'jobnameprefix':sub})
    subs_run.append(sub)

with open(f'.githash/{timestamp}_githash.txt', 'a') as f:
    for sub in subs_run:
        f.write(f'\n{sub}')
    f.write('\nabove subjects submitted')
