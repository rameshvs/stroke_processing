#!/usr/bin/env bash
site=$1
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

source $DIR/../stroke.cfg 2>/dev/null
sitedir=${processed_0204}/site${site}/
output=${volume_csv_path}/site${site}.csv
echo $sitedir
echo '"Subject","WMHv"' > $output
find $sitedir -name "*seg.txt" | sort | while read f; do
    subj=$(echo $f | cut -d "/" -f 10)
    # make a csv row
    echo '"'$subj'","'$(cat $f)'"'
done >> $output
