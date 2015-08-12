#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

source $DIR/../stroke.cfg 2>/dev/null

#source ../stroke.cfg 2>/dev/null

site=$1
sitedir=${processed_0204}/site${site}/
echo $sitedir
images=$(find $sitedir -name "*wmh_threshold_seg_in_atlas.nii.gz")

cmd="$ANTSPATH/AverageImages 3 ${site_average_path}/${site}_average_wmh.nii.gz 0 $images"

echo $cmd
$cmd


