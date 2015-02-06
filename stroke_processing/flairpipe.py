#!/usr/bin/env python

# This script implements the registration pipeline described in the paper:
#
# Quantification and Analysis of Large Multimodal Clinical Image Studies:
# Application to Stroke, by Sridharan, Dalca et al.
#
# For questions, please contact {rameshvs,adalca}@csail.mit.edu.


import os
import sys

import pipebuild as pb
import tracking

cwd = os.path.dirname(os.path.abspath(__file__))

ATLAS_MODALITY = 't1'
FLAIR_INTENSITY = '290'

WMH_THRESHOLD = 430

CLOBBER_EXISTING_OUTPUTS = False

DATA_ROOT = os.path.join(pb.NFS_ROOT, 'projects/stroke')
#SCRATCH_ROOT = '/data/scratch/rameshvs'
PROCESSING_ROOT = '/data/vision/polina/projects/stroke/processed_datasets/'

#ATLAS_BASE = os.path.join(DATA_ROOT, 'work/input/atlases/flair_atlas/')
ATLAS_BASE = os.path.join(DATA_ROOT, 'work/input/atlases/')

# with open(os.path.join(DATA_ROOT, 'work', 'subject_lists', 'interrater_subj.txt')) as f:
#     SUBJECTS = f.read().splitlines()

if __name__ == '__main__':

    ########################
    ### Argument parsing ###
    ########################
    USAGE = '%s <subj> <smoothness regularization> <field regularization> <out folder> [<subj list>]' % sys.argv[0]

    if len(sys.argv) not in [5,6]:
        print(USAGE)
        sys.exit(1)

    subj = sys.argv[1]
    # Regularization parameters for ANTS
    regularization = float(sys.argv[2])
    regularization2 = float(sys.argv[3])

    # where the data lives
    data_subfolder = sys.argv[4]
    if 'site00' in data_subfolder:
        site = 'site00'
    else:
        site = 'unknown'

    if len(sys.argv) == 6:
        subject_list = open(sys.argv[5]).read().split()
        mode = 'server'
    else:
        mode = 'execute'
    #############################
    ### Set up atlas and data ###
    #############################

    BASE = os.path.join(PROCESSING_ROOT, data_subfolder)
    #BASE = os.path.join(DATA_ROOT, 'processed_datasets', data_subfolder)
    #SCRATCH_BASE = os.path.join(SCRATCH_ROOT, 'processed_datasets', data_subfolder)
    #SCRATCH_BASE = BASE
    #BASE = os.path.join('/data/vision/scratch/polina/users/rameshvs', data_subfolder)
    ## Atlas

    #atlas = pb.Dataset(ATLAS_BASE, 'buckner61{feature}{extension}', None)
    #atlas = pb.Dataset(ATLAS_BASE, 'flair_template{extension}', None)
    #atlas = pb.Dataset(ATLAS_BASE, 'flairTemplateInBuckner_sigma{kernel}{extension}', None)
    atlas = pb.Dataset(ATLAS_BASE, 'flairTemplateInBuckner_sigma{kernel}{extension}', None)
    buckner = pb.Dataset(ATLAS_BASE, 'buckner61{feature}{extension}', None)
    ## Subject data
    dataset = pb.Dataset(
                BASE,
                # How are the inputs to the pipeline stored?
                os.path.join(BASE , '{subj}/original/{modality}_1/{subj}_{modality}_{feature}'),
                # How should intermediate files be stored?
                #os.path.join(BASE, '{subj}/images/{subj}_{modality}_{feature}{modifiers}'),
                os.path.join(BASE, '{subj}/images/{subj}_{modality}_{feature}{modifiers}'),
                log_template=os.path.join(BASE, '{subj}/logs/'),
                )

    #dataset.add_mandatory_input(modality='t1', feature='raw')
    #dataset.add_mandatory_input(modality='flair', feature='img')
    dataset.add_mandatory_input(modality='flair', feature='raw')
    dataset.get_original(subj=subj, modality='t1', feature='raw')

    if mode == 'server':
        tracking.run_server(subject_list, dataset)
        sys.exit(0)
    else:
        pass
    #############################
    ### Registration pipeline ###
    #############################

    ###
    if True:#site in ['site00', 'site13', 'site18']:
        modifiers = '_prep_pad'
        first_step = pb.PyPadCommand(
                     "Pad flair by convention",
                     #cmdName=os.path.join(cwd, 'strip_header.py'),
                     input=dataset.get_original(subj=subj, modality='flair', feature='raw'),
                     output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers),
                     out_mask=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers + '_mask_seg'),
                     )

    else:
        raise NotImplementedError

    mask = dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers+'_brainmask')
    robex = pb.RobexCommand(
            "Brain extraction with ROBEX",
            input=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers),
            output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers+'_robex'),
            out_mask=mask)

    masker = pb.NiiToolsMaskCommand(
            "Apply mask from robex",
            input=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers),
            mask=mask,
            output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers+'_brain'),
            )

    modifiers += '_brain'
    # intensity_corr = pb.MCCMatchWMCommand(
    #         "Intensity correction for flair image",
    #         inFile=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers),
    #         maskFile=mask,
    #         intensity=FLAIR_INTENSITY,
    #         output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers + '_matchwm'))
    intensity_corr = pb.NiiToolsMatchIntensityCommand(
            "Intensity correction for flair image",
            inFile=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers),
            maskFile=mask,
            intensity=FLAIR_INTENSITY,
            output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers + '_matchwm'),
            clobber=True,
            )

    modifiers += '_matchwm'


    subj_final_img = dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers)
    #for sigma in [8,3,1,20]:
    for sigma in [8]:
        atlas_img = atlas.get_original(kernel=sigma)
        ###### Final atlas -> subject registration
        forward_reg = pb.ANTSCommand("Register label-blurred flair atlas with sigma %d to subject" % sigma,
                moving=atlas_img,
                fixed=subj_final_img,
                output_folder=os.path.join(dataset.get_folder(subj=subj), 'reg'),
                metric='CC',
                radiusBins=4,
                mask=mask,
                regularization='Gauss[%0.3f,%0.3f]' % (regularization,regularization2),
                method='201x201x201',
                )

        pb.ANTSWarpCommand.make_from_registration(
                "Warp subject image to atlas space using sigma %d warp" % sigma,
                moving=subj_final_img,
                reference=atlas_img,
                registration=forward_reg,
                inversion='inverse',
                clobber=True)

        label_warp = pb.ANTSWarpCommand.make_from_registration(
                "Warp atlas labels to subject space using sigma %d warp" % sigma,
                moving=buckner.get_original(feature='_seg'),
                reference=subj_final_img,
                registration=forward_reg,
                useNN=True)
        pb.ANTSWarpCommand.make_from_registration(
                "Warp atlas image to subject space using sigma %d warp" % sigma,
                moving=atlas_img,
                reference=subj_final_img,
                registration=forward_reg)

        threshold_segmentation = pb.NiiToolsMaskedThresholdCommand(
                "Threshold segmentation",
                infile=intensity_corr.outfiles[0],
                threshold=WMH_THRESHOLD,
                output=dataset.get(subj=subj, modality='flair', feature='wmh_raw_threshold_seg', modifiers=''),
                label=label_warp.outfiles[0],
                direction='greater',
                labels=[2,41],
                )


        filename = os.path.basename(label_warp.outfiles[0]).split('.')[0]
        subj_png_filename = dataset.get(subj=subj, modality='other', feature=filename, modifiers='', extension='.png')
        pb.PyFunctionCommand(
                "Generate flair with buckner label overlay",
                "tools.better_overlay",
                [subj_final_img,
                label_warp.outfiles[0],
                [15, 17, 19, 20, 21, 22, 23, 25],
                subj_png_filename],
                output_positions=[3])

        # ####### run the registration the other way
        # for mask_feature in ['_ventricles_dilate_seg', '_fixed_mask_from_seg_binary']:
        #     inverse_reg = pb.ANTSCommand("Register subject to label-blurred flair atlas with sigma %d" % sigma,
        #             fixed=atlas_img,
        #             moving=subj_final_img,
        #             output_folder=os.path.join(dataset.get_folder(subj=subj), 'reg'),
        #             metric='CC',
        #             radiusBins=4,
        #             mask=buckner.get_original(feature=mask_feature),
        #             regularization='Gauss[%0.3f,%0.3f]' % (regularization,regularization2),
        #             method='200x200x200',
        #             )

        #     subj_to_atlas_warp = pb.ANTSWarpCommand.make_from_registration(
        #             "Warp subject image to atlas space using sigma %d warp" % sigma,
        #             moving=subj_final_img,
        #             reference=atlas_img,
        #             registration=inverse_reg,
        #             inversion='forward',
        #             )
        #     label_warp = pb.ANTSWarpCommand.make_from_registration(
        #             "Warp atlas labels to subject space using sigma %d warp" % sigma,
        #             moving=buckner.get_original(feature='_seg'),
        #             reference=subj_final_img,
        #             registration=inverse_reg,
        #             inversion='inverse',
        #             useNN=True,
        #             )
        #     filename = os.path.basename(label_warp.outfiles[0]).split('.')[0]
        #     subj_png_filename = dataset.get(subj=subj, modality='other', feature=filename, modifiers='', extension='.png')
        #     pb.PyFunctionCommand(
        #             "Generate flair with buckner label overlay",
        #             "tools.better_overlay",
        #             [subj_final_img,
        #             label_warp.outfiles[0],
        #             [15, 17, 19, 20, 21, 22, 23, 25],
        #             subj_png_filename],
        #             output_positions=[3])

        # pb.ANTSWarpCommand.make_from_registration(
        #         "Warp atlas image to subject space using sigma %d warp" % sigma,
        #         moving=atlas_img,
        #         reference=subj_final_img,
        #         registration=inverse_reg,
        #         inversion='inverse',
        #         )

    for path in [os.path.join(BASE,subj,'images'),
            os.path.join(BASE,subj,'images','reg'),
            dataset.get_log_folder(subj=subj)]:
        try:
            os.mkdir(path)
        except:
            pass

    ### Generate script file and SGE qsub file
    tracker = tracking.Tracker(pb.Command.all_commands, pb.Dataset.all_datasets)
    tracker.compute_dependencies()

    ###
    # NIPYPE_ROOT = '/data/scratch/rameshvs/sandbox/nipype_regpipe'
    # wf = tracker.make_nipype_pipeline(NIPYPE_ROOT)
    log_folder = dataset.get_log_folder(subj=subj)
    pb.Command.generate_code_from_datasets([dataset, atlas], log_folder, subj, sge=True,
            wait_time=0, tracker=tracker)

