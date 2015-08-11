import os
import sys

import numpy as np
import nibabel as nib

from os.path import join as j
PROCESSING_ROOT = '/data/vision/polina/projects/stroke/processed_datasets/'

def gather_imgs_in_flair(site):
    site_dir = j(PROCESSING_ROOT, site)
    subjects = os.listdir(site_dir)
    files = []
    good_subjs = []
    for subj in sorted(subjects):
        filename = '10543_flair_img_prep_pad_brain_matchwm_IN---_flairTemplateInBuckner_sigma8_TO_NONLINEAR_GAUSS_9000_0200__201x201x201_CC4_MASKED_10543_flair_img_prep_pad_brainmask.nii.gz__10543_flair_img_prep_pad_brain_matchwm---_flairTemplateInBuckner_sigma8-.nii.gz'.replace('10543', subj)
        filename2 = '10543_flair_img_prep_pad_brain_matchwm_IN---_flairTemplateInBuckner_sigma8_TO_NONLINEAR_GAUSS_9000_0200__201x201x201_CC4_MASKED_10543_flair_img_prep_pad_brainmask.nii.gz__10543_flair_img_prep_pad_brain_matchwm----_flairTemplateInBuckner_sigma8-.nii.gz'.replace('10543', subj)
        filename = j(site_dir, subj, 'images', filename)
        filename2 = j(site_dir, subj, 'images', filename2)
        if os.path.exists(filename):
            files.append(filename)
        elif os.path.exists(filename2):
            files.append(filename2)
        else:
            continue
        good_subjs.append(subj)
    return (good_subjs, files)


def do_median(files):
    all_imgs = []
    for f in files:
        d = nib.load(f)
        all_imgs.append(d.get_data().squeeze()[...,np.newaxis])
    all_img_array = np.concatenate(all_imgs, axis=3)

    median_img = np.median(all_img_array, 3)
    measure = ((all_img_array - median_img)**2).sum(2).sum(1).sum(0)
    assert measure.size == len(files)
    return measure
if __name__ == '__main__':
    (subjs, files) = gather_imgs_in_flair(sys.argv[1])
    measures = do_median(files)

