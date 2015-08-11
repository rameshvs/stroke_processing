#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

site=$1
subjectlist=/data/vision/polina/projects/stroke/work/subject_lists/sites/${site}.txt
export PYTHONPATH=/data/vision/polina/users/rameshvs/mip4/:$DIR/../:$PYTHONPATH
echo $subjectlist

cat $subjectlist | while read subj; do
    echo $subj
    cmd="python $DIR/../stroke_processing/flairpipe_test.py $subj 9.0 0.2 2015_02_04/${site}/"
    echo $cmd
    $cmd
done
