#!/usr/bin/env python
from __future__ import print_function
import ast
import csv
import numpy as np

#SITES = [3,4,6,7,16,18,19,20,21,23]
SITES = [3,4,6,7,13,16,18,20,21]
id_field_possibilities = set()
age_field_possibilities = set()
def process_site(site):
    global id_field_possibilities
    global age_field_possibilities
    wmh_csv = "/data/vision/polina/projects/stroke/work/rameshvs/volumes/site%02d.csv" % site
    demographic_csv = '/data/vision/polina/projects/stroke/received_work/site_demographics_csv/site%d.csv' % site

    # load WMH
    with open(wmh_csv) as c:
        wmh_data = []
        subj_ids = []
        reader = csv.reader(c)
        reader.next()
        for (subj, wmhv) in reader:
            subj_ids.append(subj)
            wmh_data.append(ast.literal_eval(wmhv))

    print("I found %d subjects with WMH segmentations for site %02d" % (len(wmh_data), site))
    # load ages
    with open(demographic_csv) as c:
        ages = {}
        reader = csv.DictReader(c)
        fields = reader.fieldnames
        (id_field,) = [f for f in fields if 'id' in f.lower()]
        (age_field,) = [f for f in fields if 'age' in f.lower()]
        id_field_possibilities.add(id_field)
        age_field_possibilities.add(age_field)
        for row in reader:
            age_str = row[age_field]
            if age_str is None:
                continue
            if age_str.endswith('90'):
                age = 90 # ">= 90 just maps to 90"
            else:
                age = ast.literal_eval(age_str)
            ages[row[id_field]] = age

    age_data = []
    for subj in subj_ids:
        try:
            age_data.append(ages[subj])
        except KeyError:
            print("Warning: no data for %s" % subj)
            age_data.append(-1)

    if len(wmh_data) > 0:
        good_ages, good_wmhvs = zip(*[(a, w) for (a, w) in zip(age_data, wmh_data) if a != -1])
    else:
        good_ages = good_wmhvs = []
    return (subj_ids, good_ages, good_wmhvs)

def main():
    all_data = {}
    for site in SITES:
        (subjects, ages, wmhs) = process_site(site)
        all_data[site] = [subjects, np.array(ages), np.array(wmhs)]

    np.savez('/data/vision/polina/projects/stroke/work/rameshvs/volumes/all_data_tensites.npz', all_data=all_data)

if __name__ == '__main__':
    main()
