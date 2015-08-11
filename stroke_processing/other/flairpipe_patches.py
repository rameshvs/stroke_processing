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
FLAIR_MAX_VALUE = 1100

CLOBBER_EXISTING_OUTPUTS = False

DATA_ROOT = os.path.join(pb.NFS_ROOT, 'projects/stroke')
SCRATCH_ROOT = '/data/scratch/rameshvs'

#ATLAS_BASE = os.path.join(DATA_ROOT, 'work/input/atlases/flair_atlas/')
ATLAS_BASE = os.path.join(DATA_ROOT, 'work/input/atlases/')

# with open(os.path.join(DATA_ROOT, 'work', 'subject_lists', 'interrater_subj.txt')) as f:
#     SUBJECTS = f.read().splitlines()

if __name__ == '__main__':

    ########################
    ### Argument parsing ###
    ########################
    USAGE = '%s <subj> <smoothness regularization> <field regularization> <out folder> [scale=6>]' % sys.argv[0]

    if len(sys.argv) not in [5,6,7]:
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

    if len(sys.argv) >= 6:
        scale = sys.argv[5]
    else:
        scale = '6'

    if len(sys.argv) >= 7:
        clobber_patches_reg=False
    else:
        clobber_patches_reg = True
    #############################
    ### Set up atlas and data ###
    #############################

    BASE = os.path.join(SCRATCH_ROOT, 'stroke', data_subfolder)
    #BASE = os.path.join(DATA_ROOT, 'processed_datasets', data_subfolder)
    #SCRATCH_BASE = os.path.join(SCRATCH_ROOT, 'processed_datasets', data_subfolder)
    SCRATCH_BASE = BASE
    #BASE = os.path.join('/data/vision/scratch/polina/users/rameshvs', data_subfolder)
    ## Atlas

    #atlas = pb.Dataset(ATLAS_BASE, 'buckner61{feature}{extension}', None)
    #atlas = pb.Dataset(ATLAS_BASE, 'flair_template{extension}', None)
    #atlas = pb.Dataset(ATLAS_BASE, 'flairTemplateInBuckner_sigma{kernel}{extension}', None)
    atlas = pb.Dataset(ATLAS_BASE, 'flairTemplateInBuckner_sigma{kernel}{extension}', None)
    buckner = pb.Dataset(ATLAS_BASE, 'buckner61{feature}{extension}', None)
    ## Subject data
    dataset = pb.Dataset(
                SCRATCH_BASE,
                # How are the inputs to the pipeline stored?
                os.path.join(SCRATCH_BASE , '{subj}/original/{modality}_1/{subj}_{modality}_{feature}'),
                # How should intermediate files be stored?
                #os.path.join(BASE, '{subj}/images/{subj}_{modality}_{feature}{modifiers}'),
                os.path.join(SCRATCH_BASE, '{subj}/images/{subj}_{modality}_{feature}{modifiers}'),
                log_template=os.path.join(SCRATCH_BASE, '{subj}/logs/'),
                )

    #dataset.add_mandatory_input(modality='t1', feature='raw')
    #dataset.add_mandatory_input(modality='flair', feature='img')
    dataset.add_mandatory_input(modality='flair', feature='raw')
    dataset.get_original(subj=subj, modality='t1', feature='raw')

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
    intensity_corr = pb.MCCMatchWMCommand(
            "Intensity correction for flair image",
            inFile=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers),
            maskFile=mask,
            intensity=FLAIR_INTENSITY,
            output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers + '_matchwm'))

    modifiers += '_matchwm'

    nlm = pb.MCCInputOutputCommand("NLM superresolution on flair image",
            matlabName='niiNLMUpsample2',
            input=dataset.get(subj=subj, modality='flair', feature='img', modifiers='_prep_pad_brain_matchwm'),
            output=dataset.get(subj=subj, modality='flair', feature='img', modifiers='_prep_pad_nlm'))



    registration_categories = ['patches', 'low_resolution', 'linear_interpolation',
            'nlm']
    registrations = {}
    label_transforms = {}
    label_transforms_affine = {}
    subj_final_img = dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers)
    #for sigma in [8,3,1,20]:
    for sigma in [8]:
        original_atlas_img = atlas.get_original(kernel=sigma)
        ###### Final atlas -> subject registration
        registration_parameters = dict(
                metric='CC',
                radiusBins=4,
                regularization='Gauss[%0.3f,%0.3f]' % (regularization,regularization2),
                method='nonlinear',
                nonlinear_iterations='30x90x21',
                )
        forward_reg = pb.ANTSCommand("Register label-blurred flair atlas with sigma %d to subject" % sigma,
                moving=original_atlas_img,
                fixed=subj_final_img,
                output_folder=os.path.join(dataset.get_folder(subj=subj), 'reg'),
                mask=mask,
                **registration_parameters
                )
        registrations['low_resolution'] = forward_reg

        #for mod in [modifiers, '_prep_pad']:
        mod = modifiers
        for ratio in xrange(1,7):
            upsampler = pb.NiiToolsUpsampleCommand(
                    "Upsample image with mask and linear interp",
                    input=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers),
                    output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers + '_upsample_%d'%ratio),
                    out_mask=dataset.get(subj=subj, modality='flair', feature='img', modifiers=modifiers + '_upsample_%d_slicemask'%ratio),
                    ratio=ratio
                    )
            atlas_img = atlas.get_original(kernel = str(sigma) + '_pad_' + str(ratio))
            #atlas_img = atlas.get_original(kernel= '_pad_' + str(ratio))
            affinexfm = pb.ANTSWarpCommand("Affinely transform upsampled img to atlas",
                moving=dataset.get(subj=subj, modality='flair', feature='img', modifiers=mod + '_upsample_%d'%ratio),
                output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=mod + '_upsample_%d_affine_new'%ratio),
                reference=atlas_img,
                transforms = ' -i ' + forward_reg.affine,
                )
            pb.NiiToolsConvertTypeCommand(
                    "Convert to uint8",
                    input=dataset.get(subj=subj, modality='flair', feature='img', modifiers=mod + '_upsample_%d_affine_new'%ratio),
                    output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=mod + '_upsample_%d_affine_new_uint8'%ratio),
                    type='uint8',
                    normalization=FLAIR_MAX_VALUE,
                    )
            affinexfm_mask = pb.ANTSWarpCommand("Affinely transform upsampled mask to atlas",
                moving=dataset.get(subj=subj, modality='flair', feature='img', modifiers=mod + '_upsample_%d_slicemask'%ratio),
                output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=mod + '_upsample_%d_slicemask_affine_new'%ratio),
                reference=atlas_img,
                transforms = ' -i ' + forward_reg.affine,
                )
            pb.NiiToolsConvertTypeCommand(
                    "Convert to uint8",
                    input=dataset.get(subj=subj, modality='flair', feature='img', modifiers=mod + '_upsample_%d_slicemask_affine_new'%ratio),
                    output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=mod + '_upsample_%d_slicemask_affine_new_uint8'%ratio),
                    type='uint8',
                    normalization='prob',
                    )


        registration_subject_images = {
                'low_resolution': subj_final_img,
                'nlm': dataset.get(subj=subj, modality='flair', feature='img', modifiers='_prep_pad_nlm'),
                # TODO change me when stroke patch results are ready
                'patches': '/data/vision/polina/projects/stroke/work/adalca/patchSynthesis/stroke/stroke_%s_%s_iter1.nii.gz' % (subj, scale),
                'linear_interpolation': dataset.get(subj=subj, modality='flair', feature='img', modifiers=mod + '_upsample_%d_affine_new'%ratio),
                }


        if not os.path.exists(registration_subject_images['patches']):
            print("*"*70)
            print("WARNING: Couldn't find patches!")
            print(registration_subject_images['patches'])
            print("*"*70)
            registration_categories.remove('patches')

        affine_mask = dataset.get(subj=subj, modality='flair', feature='img', modifiers='_prep_pad_mask_affine')
        affinexfm = pb.ANTSWarpCommand("Affinely transform mask image to atlas",
            moving=mask,
            output=affine_mask,
            reference=original_atlas_img,
            transforms = ' -i ' + forward_reg.affine,
            )
        for category in registration_categories:
            affinexfm = pb.ANTSWarpCommand("Affinely transform %s image to atlas"%category,
                moving=registration_subject_images[category],
                output=dataset.get(subj=subj, modality='flair', feature='img', modifiers=mod + '_%s_affine_semifinal'%category),
                reference=original_atlas_img,
                transforms = ' -i ' + forward_reg.affine,
                )
            if category == 'low_resolution':
                continue
            params = registration_parameters.copy()
            if category == 'patches':
                params['mask'] = affine_mask
                params['clobber'] = clobber_patches_reg
            else:
                params['mask'] = mask
            registrations[category] = pb.ANTSCommand("Register label-blurred flair atlas with sigma %d to %s" % (sigma, category),
                    moving=original_atlas_img,
                    fixed=registration_subject_images[category],
                    output_folder=os.path.join(dataset.get_folder(subj=subj), 'reg'),
                    **params
                    )
            reg_sequence = [registrations[category]]
            inversion_sequence = ['forward']
            affine_only_sequence = [False]
            if category == 'patches':
                reg_sequence.append(forward_reg)
                inversion_sequence.append('forward')
                affine_only_sequence.append(True)
            label_transforms[category] = pb.ANTSWarpCommand.make_from_registration_sequence(
                "warp label map from atlas using %s"%category,
                moving=buckner.get_original(feature='_seg'),
                reference=dataset.get_original(subj=subj, modality='flair', feature='img', modifiers=mod),
                output_folder=dataset.get_folder(subj=subj),
                reg_sequence=reg_sequence,
                inversion_sequence=inversion_sequence,
                affine_only_sequence=affine_only_sequence,
                useNN=True,
                )

    for category in registration_categories:
        atlas_pad = buckner.get_original(feature= '_pad_7')
        
        # # this is totally wrong; don'tuse it
        # xfms_affine = registrations[category].warp + ' -i ' + \
        #         registrations[category].affine + ' ' + \
        #         registrations[category].inverse_warp

        affine_atlas_filename = dataset.get(subj=subj, modality='flair', feature='img', modifiers='_' + category + '_in_affine_atlas')
        if category != 'patches':
            affinexfm = pb.ANTSWarpCommand("Affinely transform %s to atlas" % category,
                moving=registration_subject_images[category],
                output=affine_atlas_filename,
                reference=atlas_pad,
                transforms = ' -i ' + forward_reg.affine + ' ',
                #transforms = xfms_affine,
                clobber=True,
                )
        else:
            patch_resampler = pb.NiiToolsUpsampleCommand(
                    "resample result into 1x1x1",
                    input=registration_subject_images[category],
                    output=affine_atlas_filename,
                    out_mask=dataset.get(subj=subj, modality='flair', feature='img', modifiers='tmp_mask'),
                    ratio=1.1666666666666,
                    method='linear',
                    clobber=True,
                    )
        affinexfm = pb.ANTSWarpCommand("Affinely transform %s to atlas" % category,
            moving=registration_subject_images[category],
            output=dataset.get(subj=subj, modality='flair', feature='img', modifiers='_' + category + '_in_affine_lowres_atlas'),
            reference=buckner.get_original(feature='_pad_1'),
            #transforms = xfms_affine,
            transforms = ' -i ' + forward_reg.affine + ' ',
            clobber=True,
            )
        nn_interpolator = pb.NiiToolsUpsampleCommand(
                "Upsample image with mask and linear interp",
                input=dataset.get(subj=subj, modality='flair', feature='img', modifiers='_' + category + '_in_affine_lowres_atlas'),
                output=dataset.get(subj=subj, modality='flair', feature='img', modifiers='_' + category + '_in_affine_lowres_NN_atlas'),
                out_mask=dataset.get(subj=subj, modality='flair', feature='img', modifiers='tmp_mask'),
                ratio=6,
                method='nearest',
                clobber=True,
                )

        xfms = registrations[category].affine + '  ' + \
                registrations[category].warp + ' -i ' + \
                registrations[category].affine

        label_transforms_affine[category] = pb.ANTSWarpCommand(
                "Warp label atlas using nonlinear only",
                moving=buckner.get_original(feature='_seg'),
                reference=buckner.get_original(feature= '_pad_7'),
                output=dataset.get(subj=subj, modality='flair', feature='img', modifiers='_' + category + '_in_affine_atlas_seg'),
                transforms=xfms,
                useNN=True,
                clobber=True,
                )
        label_transforms_affine[category] = pb.ANTSWarpCommand(
                "Warp label atlas using nonlinear only",
                moving=buckner.get_original(feature='_seg'),
                reference=buckner.get_original(feature= '_pad_1'),
                output=dataset.get(subj=subj, modality='flair', feature='img', modifiers='_' + category + '_in_affine_lowres_atlas_seg'),
                transforms=xfms,
                useNN=True,
                clobber=True,
                )
        nn_interpolator = pb.NiiToolsUpsampleCommand(
                "Upsample image with mask and linear interp",
                input=dataset.get(subj=subj, modality='flair', feature='img', modifiers='_' + category + '_in_affine_lowres_atlas_seg'),
                output=dataset.get(subj=subj, modality='flair', feature='img', modifiers='_' + category + '_in_affine_lowres_NN_atlas_seg'),
                out_mask=dataset.get(subj=subj, modality='flair', feature='img', modifiers='tmp_mask'),
                ratio=6,
                method='nearest',
                clobber=True,
                )

    for path in [os.path.join(SCRATCH_BASE,subj,'images'),
            os.path.join(SCRATCH_BASE,subj,'images','reg'),
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

