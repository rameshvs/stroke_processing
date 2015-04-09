#!/usr/bin/env bash

site=$1
subjectlist=/data/vision/polina/projects/stroke/work/subject_lists/sites/${site}.txt
ls /data/vision/polina/projects/stroke/processed_datasets/2015_02_04/site${site}  > $subjectlist
