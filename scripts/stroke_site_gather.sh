#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

site=$1
subjectlist=/data/vision/polina/projects/stroke/work/subject_lists/sites/${site}.txt
export PYTHONPATH=/data/vision/polina/users/rameshvs/medical-imaging-pipelines/:$DIR/../:$PYTHONPATH

echo $subjectlist
cat $subjectlist | sed -n '9p' | while read subj; do
    cmd="python $DIR/../stroke_processing/flairpipe_tracking.py $subj 9.0 0.2 2015_02_04/${site}/"
    echo $cmd
    $cmd
done
