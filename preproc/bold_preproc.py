
from os import path

import nipype.interfaces.fsl as fsl
from nipype import Function, JoinNode, MapNode, Node, Workflow
from nipype.algorithms import confounds as ni_confounds
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import IdentityInterface
from niworkflows.interfaces.confounds import NormalizeMotionParams


def get_wf_bold_preproc(experiment_dir, working_dir, output_dir):
    
    inputnode = Node(IdentityInterface(fields=['func']),
                  name="inputnode")

    outputnode = Node(IdentityInterface(fields=['func','par_file','fwd_file']),
                  name="outputnode")
    

    preproc = Workflow(name='bold_preproc')
    preproc.base_dir = path.join(experiment_dir, working_dir, output_dir)

    def getreferencevolume(func, referencetype='standard'):

        import nibabel as nib
       
        funcfile = func
        if isinstance(func, list):
            funcfile = func[0]
       
        # load functional time series
        epi_img = nib.load(funcfile)
        epi_img_data = epi_img.get_fdata()
       
        x = int(int(epi_img_data.shape[3]) / 2) - 1
        if referencetype=='standard':
            # Standard reference, just pick middle volume
            reference = x
        else:
            raise NotImplementedError(f"Reference type '{referencetype}' is not implemented.")

        return reference

    get_reference_node = Node(
        Function(
            function=getreferencevolume, 
            input_names=['func', 'referencetype'], 
            output_names=['reference']
        ), 
        name=f'getreferencevolume'
    )
    # Hard set for now, robust reference not implemented in this version
    get_reference_node.inputs.referencetype = "standard"
    
    extract_ref = Node(interface=fsl.ExtractROI(t_size=1), name='extractref')
    
    #  MCFLIRT
    motion_correct = Node(fsl.MCFLIRT(save_plots=True,output_type='NIFTI'), name="mcflirt")

    #  Plot output
    plot_motion = MapNode(
        interface=fsl.PlotMotionParams(in_source='fsl'),
        name='plot_motion',
        iterfield=['in_file'])
    plot_motion.iterables = ('plot_type', ['rotations', 'translations'])
    join_plot_motion = JoinNode(IdentityInterface(fields = ['out_file']), name='join_plot_motion', joinfield='out_file', joinsource='plot_motion')

    # Normalize motion from FSL to SPM format
    normalize_motion = Node(NormalizeMotionParams(format='FSL'),
                                name="normalize_motion")

    calc_fwd = Node(
        interface=ni_confounds.FramewiseDisplacement(parameter_source='SPM'),   # use parameter source as SPM because these have been processed with NormalizeMotionParams, for compatability with fmriprep
        name='calc_fwd'
    )

    def plot_bad_volumes(in_file, fwd, nimgs=10):
        import os

        import nibabel as nib
        import numpy as np
        from PIL import Image

        img=nib.load(in_file)
        dat = img.get_fdata()

        # Sort by fwd, worst will be at end
        fwdvals=np.loadtxt(fwd, skiprows=1)
        fwd=np.insert(fwd,0,np.nan) # fwd is a difference between two images. We'll use the fwd to index the second of these (likely more artifacts due to spin history)
        fwdind = np.argsort(fwdvals)

        out_files=[]

        # Convert 4D series into 3d imgs
        img3 = nib.funcs.four_to_three(img)

        worstfwd=[]

        # Display [nimgs] worst
        for nimg in range(nimgs):
            if nimg:
                title="worst"
            else:
                title="best"
            worstfwd.append(fwdvals[fwdind[-nimg]])
            dat=img3[fwdind[-nimg]].get_fdata()
            # Axial slices
            dat_agg=None 
            for z in range(0,dat.shape[2],3):
                slice = np.rot90(dat[:,:,z])
                if dat_agg is None:
                    dat_agg = slice
                else:
                    dat_agg = np.concatenate((dat_agg,slice), axis=1)

            dat_agg=255*dat_agg/(np.max(dat_agg))
            im = Image.fromarray(dat_agg)
            im = im.convert("L")
            out_file = os.path.abspath(f"{title}_epi_axial_{nimg}.png")
            im.save(out_file)
            out_files.append(out_file)

            # Sagittal slices
            dat_agg=None
            for x in range(0,dat.shape[0],3):
                slice = np.rot90(dat[x,:,:])
                if dat_agg is None:
                    dat_agg = slice
                else:
                    dat_agg = np.concatenate((dat_agg,slice), axis=1)

            dat_agg=255*dat_agg/(np.max(dat_agg))
            im = Image.fromarray(dat_agg)
            im = im.convert("L")
            out_file = os.path.abspath(f"{title}_epi_sagittal_{nimg}.png")
            im.save(out_file)
            out_files.append(out_file)

        with open("worst_epi_fwd.txt","w") as f:
            for item in worstfwd:
                f.write(f"{item}\n")

        return out_files, os.path.abspath("worst_epi_fwd.txt")

    plot_bad_volumes_node = Node(
        Function(
            function=plot_bad_volumes, 
            input_names=['in_file', 'fwd'], 
            output_names=['out_file', 'worst_fwd']
            ), 
        name=f'plot_bad_volumes'
        )

    # Datasink - creates output folder for important outputs
    datasink = Node(DataSink(base_directory=experiment_dir,
                            container=output_dir),
                    name="datasink")

    preproc.connect(inputnode, 'func', motion_correct,  'in_file')
    preproc.connect(inputnode, 'func', extract_ref, 'in_file')
    preproc.connect(inputnode, 'func', get_reference_node, 'func')
    preproc.connect(get_reference_node, 'reference', extract_ref, 't_min')
    preproc.connect(extract_ref, 'roi_file', motion_correct, 'ref_file')
    preproc.connect([(motion_correct, normalize_motion, [('par_file', 'in_file')])])
    preproc.connect(motion_correct, 'par_file', plot_motion, 'in_file')
    preproc.connect([(plot_motion,  join_plot_motion, [('out_file', 'out_file')])])
    preproc.connect(normalize_motion, 'out_file', calc_fwd, 'in_file')


    preproc.connect([(join_plot_motion, datasink, [('out_file', 'motion_plots')])])
    preproc.connect([(motion_correct, datasink, [('par_file', 'motion_parameters')])])
    preproc.connect([(calc_fwd, datasink, [('out_file', 'motion_fwd')] )])

    preproc.connect(inputnode, "func", plot_bad_volumes_node, "in_file")
    preproc.connect(calc_fwd, "out_file", plot_bad_volumes_node, "fwd")

    preproc.connect(plot_bad_volumes_node, "out_file", datasink, "bad_volumes")
    preproc.connect(plot_bad_volumes_node, "worst_fwd", datasink, "bad_volumes_fwd")

    preproc.connect([(motion_correct, outputnode, [('out_file', 'func'),('par_file','par_file')] )])
    preproc.connect([(calc_fwd, outputnode, [('out_file', 'fwd_file')] )])

    return preproc
