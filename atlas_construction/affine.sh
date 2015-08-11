#!/usr/bin/env bash

site=$1
export ANTSPATH=/data/vision/polina/shared_software/ANTS/build/bin/

oldpwd=$(pwd)
cd /data/vision/polina/projects/stroke/work/rameshvs/site_atlases/site${site}/tmp_affine

$ANTSPATH/buildtemplateparallel.sh -d 3 -m 1x0x0 -n 0 -t RA  -s CC -c 1 -j 5 -o ../affine ../images/*

cd $oldpwd
