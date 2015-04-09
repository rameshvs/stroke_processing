#!/usr/bin/env bash

ANTSPATH=/data/vision/polina/shared_software/ANTS/build/bin

site=$1
sitedir=/data/vision/polina/projects/stroke/processed_datasets/2015_02_04/site${site}/
echo $sitedir
images=$(find $sitedir -name "*wmh_threshold_seg_in_atlas.nii.gz")

cmd="$ANTSPATH/AverageImages 3 /data/vision/polina/projects/stroke/work/rameshvs/sites/${site}_average_wmh.nii.gz 0 $images"

echo $cmd
$cmd


