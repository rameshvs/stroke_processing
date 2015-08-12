#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

source $DIR/../stroke.cfg 2>/dev/null
site=$1
subjectlist=${site_subject_list}/${site}.txt
export PYTHONPATH=${pipebuilder_path}:$DIR/../:$PYTHONPATH

echo $subjectlist
cat $subjectlist | sed -n '9p' | while read subj; do
    cmd="python $DIR/../stroke_processing/flairpipe_tracking.py $subj 9.0 0.2 2015_02_04/${site}/"
    echo $cmd
    $cmd
done
