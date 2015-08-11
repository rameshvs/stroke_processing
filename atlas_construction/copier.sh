#!/usr/bin/env bash

site=$1
#infolder=/data/vision/polina/projects/ADNI/work/adalca/data-brains/copy_over2csail/all/
strokedir=/data/vision/polina/projects/stroke/processed_datasets/2015_02_04/site${site}/
workdir=/data/vision/polina/projects/stroke/work/rameshvs/site_atlases/site${site}/
ls $strokedir
for subj in $(ls $strokedir); do
    cp ${strokedir}/${subj}/original/flair_1/${subj}_flair_raw.nii.gz ${workdir}/images/
done
