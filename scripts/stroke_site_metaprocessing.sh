#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

site=$1
subjectlist=/data/vision/polina/projects/stroke/work/subject_lists/sites/${site}.txt
export PYTHONPATH=/data/vision/polina/users/rameshvs/medical-imaging-pipelines/:$DIR/../:$PYTHONPATH

cat $subjectlist | while read subj; do
    python $DIR/../stroke_processing/flairpipe_tracking.py $subj 9.0 0.2 2015_02_04/${site}/
done
