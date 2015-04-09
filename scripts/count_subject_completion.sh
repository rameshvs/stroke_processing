#!/usr/bin/env bash

oldpwd=$(pwd)

cd /data/vision/polina/projects/stroke
for siteno in 03 06 07 16 18 19 20 21 23; do
    total_subjs=$(ls raw_datasets/${siteno} | wc -l);
    flair_count=$(find processed_datasets/2015_02_04/site${siteno}/ -name "*.nii.gz" | grep flair_1 | wc -l);
    regcount=$(find processed_datasets/2015_02_04/site${siteno}/ -name "buckner61_seg_IN---_flairTemplateInBuckner_sigma8_TO_NONLINEAR_GAUSS_9000_0200__201x201x201_CC4_MASKED_*" | wc -l);
    echo "$total_subjs $regcount";
done > work/rameshvs/counts_0205.txt

cd $oldpwd
