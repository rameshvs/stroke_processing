#!/usr/bin/env bash
site=$1
sitedir=/data/vision/polina/projects/stroke/processed_datasets/2015_02_04/site${site}/
output=/data/vision/polina/projects/stroke/work/rameshvs/volumes/site${site}.csv
echo $sitedir
echo '"Subject","WMHv"' > $output
find $sitedir -name "*seg.txt" | sort | while read f; do
    subj=$(echo $f | cut -d "/" -f 10)
    # make a csv row
    echo '"'$subj'","'$(cat $f)'"'
done >> $output
