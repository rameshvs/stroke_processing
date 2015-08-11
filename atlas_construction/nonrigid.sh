#!/usr/bin/env bash
set -e

site=$1

export ANTSPATH=/data/vision/polina/shared_software/ANTS/build/bin/

oldpwd=$(pwd)
cd /data/vision/polina/projects/stroke/work/rameshvs/site_atlases/site${site}/tmp_nonrigid

$ANTSPATH/buildtemplateparallel.sh -d 3 -m 30x50x20 -n 0 -t GR  -s CC -c 1 -j 5 -o ../site${site}_template_ -z ../affinetemplate.nii.gz  ../images/*

cd $oldpwd
