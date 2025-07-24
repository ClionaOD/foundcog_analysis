from os import path

import pandas as pd
from bids.layout import BIDSLayout
from nipype import config
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.pipeline.engine import Node, Workflow

from analysis.glm import model_run, single_sub_betas

config.enable_debug_mode()

############## 

experiment_dir = path.abspath(path.join("/foundcog", "dataset_sharing"))
database_path = path.abspath(path.join(experiment_dir, "bidsdatabase"))
output_dir = path.join("derivatives", "foundcog_glm")  # will be linked with experiment_dir in DataSink

layout = BIDSLayout(experiment_dir, database_path=database_path)

# list of subject identifiers
subject_list = layout.get_subjects()

## Exclude 9022 because TRs/syncing messed up
exclude_subs = ['9022']
subject_list = [i for i in subject_list if not i in exclude_subs]

subject_list.sort()

print(f'Subjects {subject_list}')

# TR as recorded by regression of 's' onset recordings
TR=0.610
fwd_cutoff = 1.5

for sub in subject_list:
    if sub != '2001':
        continue

    for task in ['pictures']:
        for reps,eg in zip([True,False],[False,True]):
            if task == 'videos' and eg:
                continue
            
            model_type = 'reps' if reps else 'eg'
            
            # NiPype setup
            working_dir = path.join("workingdir",sub)

            glm = Workflow(name='foundcog_glm')
            glm.base_dir = path.join(experiment_dir, working_dir)
            
            # Infosource - a function free node to iterate over our items
            infosource = Node(IdentityInterface(fields=['sub','task']), name="infosource")
            infosource.iterables = [('sub', [sub]),('task',[task])]

            # firstly create the node to model each run for this participant and task
            modelrun = Node(Function(input_names=['sub','task','recorded_tr','brain_mask','fwd_cutoff','rep_marking','exemplar_marking', 'derivs'],
                                        output_names=['outpaths','skippath','rep_marking', 'exemplar_marking'],
                                        function=model_run),
                                    name='modelrun')
            
            modelrun.inputs.recorded_tr = TR
            modelrun.inputs.brain_mask = '/foundcog/templates/mask/nihpd_asym_08-11_fcgmask_2mm.nii.gz' if sub[0]=='9' else '/foundcog/templates/mask/nihpd_asym_02-05_fcgmask_2mm.nii.gz'
            modelrun.inputs.fwd_cutoff = fwd_cutoff
            modelrun.inputs.derivs = 'normalized_to_common_space'
            
            modelrun.inputs.rep_marking = reps 
            modelrun.inputs.exemplar_marking = eg
            
            glm.connect([(infosource, modelrun, [('sub','sub'),('task','task')])])

            # build function with mini workflow to save out files where files have been created
            # using this convention because not all runs will pass through our various thresholds (motion etc.)
            def model_to_save(outpaths,save_dir,output_dir):
                from nipype.interfaces.io import DataSink

                # check firstly if there are any runs with model paths saved out
                if len(outpaths) != 0:
                    for run in outpaths:
                        datasink = DataSink(base_directory=save_dir, container=output_dir)
                        datasink.inputs.models = run
                        datasink.inputs.substitutions = [('_sub_','sub-'),('task_','task-')]
                        # just do this single threaded as it's within the wider slurm job
                        datasink.run()
                else:
                    print('This subject has no models saved')
            model_tosave = Node(Function(input_names=['outpaths','save_dir','output_dir'],
                        function=model_to_save), name='model_tosave')
            model_tosave.inputs.save_dir = experiment_dir
            model_tosave.inputs.output_dir = output_dir

            glm.connect([(modelrun, model_tosave, [('outpaths','outpaths')])])
            
            # save out matrix of betas for this participant
            # need to get input into subbetas from outpaths of modelrun
            subbetas = Node(Function(input_names=['sub','task','subrunpaths','rep_marking','exemplar_marking'],
                                        output_names=['outpaths'],
                                        function=single_sub_betas),
                                    name='subbetas')
            
            glm.connect([(infosource, subbetas, [('sub','sub'),('task','task')])])
            glm.connect([(modelrun, subbetas, [('outpaths','subrunpaths')])])
            glm.connect([(modelrun, subbetas, [('rep_marking','rep_marking'),('exemplar_marking','exemplar_marking')])])
            
            # saving betas now
            def beta_to_save(outpaths,save_dir,output_dir):
                from nipype.interfaces.io import DataSink
                from nipype.pipeline.engine import Node

                # check firstly if there are any runs with model paths saved out
                if len(outpaths) != 0:
                    for run in outpaths:
                        datasink = DataSink(base_directory=save_dir, container=output_dir)
                        datasink.inputs.betas = run
                        datasink.inputs.substitutions = [('_sub_','sub-'),('task_','task-')]
                        # just do this single threaded as it's within the wider slurm job
                        datasink.run()
                else:
                    print('This subject has no betas saved')
            beta_tosave = Node(Function(input_names=['outpaths','save_dir','output_dir'],
                        function=beta_to_save), name='beta_tosave')
            beta_tosave.inputs.save_dir = experiment_dir
            beta_tosave.inputs.output_dir = output_dir

            glm.connect([(subbetas, beta_tosave, [('outpaths','outpaths')])])
            
            # glm.run()
            glm.run(plugin='SLURMGraph', plugin_args = {'dont_resubmit_completed_jobs': False})
