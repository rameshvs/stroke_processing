#!/usr/bin/env bash

site=$1
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

source $DIR/../stroke.cfg 2>/dev/null
subjectlist=${site_list_path}/${site}.txt
export PYTHONPATH=${pipebuilder_path}:$DIR/../:$PYTHONPATH
echo $subjectlist

cat $subjectlist | head -n 1| while read subj; do
    echo $subj
    python $DIR/../stroke_processing/registration/flairpipe.py $subj 9.0 0.2 2015_02_04/${site}/
done
