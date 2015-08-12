#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source $DIR/../stroke.cfg 2>/dev/null
site=$1
subjectlist=${site_subject_lists}/${site}.txt
ls ${processed_0204}/site${site}  > $subjectlist
