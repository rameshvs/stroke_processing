#!/usr/bin/env bash

ANTSPATH=/data/vision/polina/shared_software/ANTS/build/bin

site=$1
sitedir=/data/vision/polina/projects/stroke/processed_datasets/2015_02_04/site${site}/
tipixdir=/afs/csail.mit.edu/u/${USER:0:1}/$USER/public_html/tipix/stroke/registration_0211/site${site}
mkdir -p $tipixdir
images=$(find $sitedir -name "*other_buckner_labels.png" | sort)

i=1
for img in $images; do
    cp $img $tipixdir/${i}.png
    i=$((i+1))
done


