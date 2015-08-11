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
DWI_INTENSITY = '210'

WMH_THRESHOLD = 430
STROKE_THRESHOLD = 1
STROKE_THRESHOLD = 1

CLOBBER_EXISTING_OUTPUTS = False

DATA_ROOT = os.path.join(pb.NFS_ROOT, 'projects/stroke')
#SCRATCH_ROOT = '/data/scratch/rameshvs'
PROCESSING_ROOT = '/data/vision/polina/projects/stroke/processed_datasets/'

#ATLAS_BASE = os.path.join(DATA_ROOT, 'work/input/atlases/flair_atlas/')
ATLAS_BASE = os.path.join(DATA_ROOT, 'work/input/atlases/')

# with open(os.path.join(DATA_ROOT, 'work', 'subject_lists', 'interrater_subj.txt')) as f:
#     SUBJECTS = f.read().splitlines()

def check_fluid_attenuation(input_filename, seg_filename, output_filename):
    import nibabel as nib
    import numpy as np
    data = nib.load(input_filename).get_data()
    seg = nib.load(seg_filename).get_data()
    ventricle = np.median(data[np.logical_and(seg==4, seg==43)])
    wm = np.median(data[np.logical_and(seg==2, seg==41)])
    with open(output_filename, 'w') as f:
        if ventricle >= wm:
            f.write('1')
        else:
            f.write('0')
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
    flair_input = dataset.get_original(subj=subj, modality='flair', feature='raw')
    dwi_input = dataset.get_original(subj=subj, modality='dwi', feature='raw')
    if True:#site in ['site00', 'site13', 'site18']:
        modifiers = '_prep_pad'
        first_step = pb.PyPadCommand(
                     "Pad flair by convention",
                     #cmdName=os.path.join(cwd, 'strip_header.py'),
                     input=flair_input,
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
            )

    modifiers += '_matchwm'


    subj_final_img = dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers)

    dwi_mask = dataset.get(subj=subj, modality='flair', feature='mask', modifiers='_from_flair')
    for sigma in [8]:
        atlas_img = atlas.get_original(kernel=sigma)
        basic_threshold_segmentation_wmh = dataset.get(subj=subj, modality='flair', feature='wmh_raw_threshold_seg', modifiers='')
        basic_threshold_segmentation_stroke = dataset.get(subj=subj, modality='flair', feature='wmh_raw_threshold_seg', modifiers='')
        multimodal_registration = pb.ANTSCommand("Rigidly register DWI to FLAIR",
                moving=dwi_input,
                fixed=subj_final_img,
                output_folder=os.path.join(dataset.get_folder(subj=subj), 'reg'),
                metric='MI',
                radiusBins=32,
                mask=mask,
                method='rigid',
                )
        pb.ANTSWarpCommand.make_from_registration(
                "Warp mask to DWI",
                moving=mask,
                reference=dwi_input,
                output_filename=dwi_mask,
                registration=multimodal_registration,
                inversion='forward'
                )

        ###### Final atlas -> subject registration
        forward_reg = pb.ANTSCommand(
                "Register label-blurred flair atlas  to subject",
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
                "Warp subject image to atlas space using  warp",
                moving=subj_final_img,
                reference=atlas_img,
                output_filename=dataset.get(subj=subj, modality='flair', feature='img', modifiers='_in_atlas'),
                registration=forward_reg,
                inversion='inverse',
                )

        label_warp = pb.ANTSWarpCommand.make_from_registration(
                "Warp atlas labels to subject space using  warp",
                moving=buckner.get_original(feature='_seg'),
                reference=subj_final_img,
                registration=forward_reg,
                useNN=True,
                )
        dwi_seg = dataset.get(subj=subj, modality='dwi', feature='seg', modifiers='')
        dwi_is_dwi = dataset.get(subj=subj, modality='dwi', feature='verified', modifiers='', extension='.txt')
        label_warp_dwi = pb.ANTSWarpCommand.make_from_registration_sequence(
                "Warp atlas labels to dwi",
                moving=buckner.get_original(feature='_seg'),
                reference=dwi_input,
                output_filename=dwi_seg,
                reg_sequence=[forward_reg, multimodal_registration],
                inversion_sequence=['forward', 'inverse'],
                useNN=True,
                )
        pb.PyFunctionCommand(
                "Verify ventricles greater than white matter",
                function="flairpipe.check_fluid_attenuation",
                args=[
                    dwi_input,
                    dwi_seg,
                    dwi_is_dwi
                    ],
                output_positions=[2])

        dwi_matchwm = dataset.get(subj=subj, modality='dwi', feature='img', modifiers='_matchwm')
        intensity_corr_dwi = pb.NiiToolsMatchIntensityCommand(
                "Intensity correction for DWI image",
                inFile=dwi_input,
                maskFile=dwi_mask,
                intensity=DWI_INTENSITY,
                output=dwi_matchwm,
                )
        pb.ANTSWarpCommand.make_from_registration(
                "Warp atlas image to subject space using  warp",
                moving=atlas_img,
                reference=subj_final_img,
                output_filename=dataset.get(subj=subj, modality='atlas', feature='img', modifiers='_in_subject'),
                registration=forward_reg)

        # threshold_segmentation_dwi = pb.NiiToolsMaskedThresholdCommand(
        #         "Threshold segmentation for stroke",
        #         infile=dwi_matchwm,
        #         threshold=STROKE_THRESHOLD,
        #         output=basic_threshold_segmentation_stroke,
        #         label=dwi_seg,
        #         direction='greater',
        #         labels=[2,41],
        #         )

        # threshold_segmentation_dwi_count = pb.NiiToolsMaskedThresholdCountCommand(
        #         "Threshold segmentation for stroke computation",
        #         infile=dwi_matchwm,
        #         threshold=STROKE_THRESHOLD,
        #         output=dataset.get(subj=subj, modality='other', feature='stroke_raw_threshold_seg', modifiers='', extension='.txt'),
        #         label=dwi_seg,
        #         direction='greater',
        #         units='mm',
        #         labels=[2,41],
        #         )
        threshold_segmentation = pb.NiiToolsMaskedThresholdCommand(
                "Threshold segmentation",
                infile=intensity_corr.outfiles[0],
                threshold=WMH_THRESHOLD,
                output=basic_threshold_segmentation_wmh,
                label=label_warp.outfiles[0],
                direction='greater',
                labels=[2,41],
                )

        threshold_segmentation_count = pb.NiiToolsMaskedThresholdCountCommand(
                "Threshold segmentation computation",
                infile=intensity_corr.outfiles[0],
                threshold=WMH_THRESHOLD,
                output=dataset.get(subj=subj, modality='other', feature='wmh_raw_threshold_seg', modifiers='', extension='.txt'),
                label=label_warp.outfiles[0],
                direction='greater',
                units='mm',
                labels=[2,41],
                )

        threshold_seg_to_atlas = pb.ANTSWarpCommand.make_from_registration(
                "Warp threshold segmentation to atlas space",
                moving=basic_threshold_segmentation_wmh,
                reference=atlas_img,
                registration=forward_reg,
                output_filename=dataset.get(subj=subj, modality='wmh_threshold_seg', feature='in_atlas', modifiers=''),
                inversion='inverse')

        filename = os.path.basename(label_warp.outfiles[0]).split('.')[0]
        #subj_png_filename = dataset.get(subj=subj, modality='other', feature=filename, modifiers='', extension='.png')
        subj_png_filename = dataset.get(subj=subj, modality='other', feature='buckner_labels', modifiers='', extension='.png')
        pb.PyFunctionCommand(
                "Generate flair with buckner label overlay",
                "tools.better_overlay",
                [subj_final_img,
                label_warp.outfiles[0],
                [15, 17, 19, 20, 21, 22, 23, 25],
                subj_png_filename],
                output_positions=[3],
                )

        # ####### run the registration the other way
        # for mask_feature in ['_ventricles_dilate_seg', '_fixed_mask_from_seg_binary']:
        #     inverse_reg = pb.ANTSCommand("Register subject to label-blurred flair atlas with ",
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
        #             "Warp subject image to atlas space using  warp",
        #             moving=subj_final_img,
        #             reference=atlas_img,
        #             registration=inverse_reg,
        #             inversion='forward',
        #             )
        #     label_warp = pb.ANTSWarpCommand.make_from_registration(
        #             "Warp atlas labels to subject space using  warp",
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
        #         "Warp atlas image to subject space using  warp",
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
    pb.Command.generate_code_from_datasets([dataset, atlas], log_folder, subj, sge=False,
            wait_time=0, tracker=tracker)
    site = 'site' + data_subfolder.split('site')[-1][:2]
    #subjects_file = '/data/vision/polina/projects/stroke/work/subject_lists/sites/%s.txt' % site
    subjects_file = '/data/vision/polina/projects/stroke/work/rameshvs/site16_completion_tmp.txt'
    subjects = open(subjects_file).read().split()
    aggregate_json = '/data/vision/polina/projects/stroke/work/rameshvs/site_pipeline_data/%s.json' % site
    # if not os.path.exists(aggregate_json):
    if True:
        tracker.gather_multi_subject_jsons(subjects, subj, aggregate_json)
    import json
    tracking.run_server(subjects, dataset, json.load(open(aggregate_json)))

