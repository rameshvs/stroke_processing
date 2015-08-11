from __future__ import print_function

import sys
import os
import shutil
from os.path import join as j
import csv
import sqlite3
import collections
import tempfile
import subprocess

import dicom
import nibabel as nib
import numpy as np


# STROKE='/data/vision/polina/projects/stroke'
# #DATA_ROOT= j(STROKE, 'tmp_oct_data/')
# #DATA_ROOT= '/data/vision/polina/users/rameshvs/test'
# DATA_ROOT= j(STROKE, 'raw_datasets', site, 'images')
# RENAMED_ROOT= j(STROKE, 'received_work', site, 'renamed')
# DATA_OUT= j(STROKE, 'processed_datasets', process_date, site)

# OUT_CSV= j(STROKE, 'work', 'rameshvs', site + '.csv')
# DB_FILE= j(STROKE, 'work', 'rameshvs', site + '.db')
# RENAMING_CSV= j(STROKE, 'work', 'rameshvs', site + '_renaming.csv')
# BINARY_CSV= j(STROKE, 'work', 'rameshvs', site + '_binary.csv')

# stores the result of manual curation: what strings map to what modalities
modality_mappings = {'GR': 'GR',
                     frozenset(('GR', 'IR')): 'IRGR',
                     frozenset(('IR', 'SE')): 'IRSE'}

modality_keywords = {
        'dwi': ['dw', 'diff', 'sshdti'],
        'flair': ['dark', 'fla'],
        't1': ['t1']}
        # 'gre': ['t2'],
        # 'tof': ['fi3d', 'fl3d', 'tof'],
        # 'adc': ['adc']}

def unsafe_csv2db(csv_filename, db_filename, field_names, table_name):
    """
    Converts a CSV file to a database.
    WARNING: vulnerable to SQL injections EVERYWHERE. don't use unless
    you trust the csv file
    """
    db = sqlite3.connect(db_filename).cursor()
    with open(csv_filename) as cf:
        reader = csv.DictReader(cf)
        # TODO fix injection vulnerability here
        db.execute('CREATE TABLE ? (%s)' % ', '.join([ f + ' text' for f in reader.fieldnames]), table_name)
        # TODO fix injection vulnerability here
        db.execute("INSERT INTO ? VALUES ")

def collapse_volumes(dwi_file):
    """
    Takes an n-volume file and selects the last volume. destructive/in-place.
    returns True if there were multiple volumes and false otherwise
    """
    nii = nib.load(dwi_file)
    data = nii.get_data()
    if len(data.shape) == 3:
        return False
    else:
        assert len(data.shape) == 4
    if data.shape[3] != 5:
        print("Found volume with %d volumes:\n%s" % (data.shape[3], dwi_file))
    new_data = data[:,:,:, -1]
    new = nib.Nifti1Image(new_data, header=nii.get_header(), affine=nii.get_affine())
    new.to_filename(dwi_file)
    return True

def clean_image(nii_file, output_file):
    """ takes a nifti file from freesurfer and saves it with an identity affine matrix """
    nii = nib.load(nii_file)
    test_matrix = np.array([[-1,0,0],[0,0,1],[0,-1,0]])
    assert np.max(np.abs(test_matrix-nii.get_affine()[:3,:3])) < 1e-4, "The freesurfer output image wasn't aligned the way I was expecting."

    new_data = nii.get_data()[-1::-1,-1::-1,:].transpose(0,2,1)
    new_affine = np.eye(4)
    new_affine[:3,3] = -(nii.get_data().shape/2).astype('int64')

    new_nii = nib.Nifti1Image(new_data,affine=new_affine,header=nii.get_header())

    new_nii.to_filename(output_file)

def check_substring(keywords, fields):
    for field in fields:
        if field is None:
            field = ''
        field = field.lower()
        for keyword in keywords:
            if keyword in field:
                return True
    return False

def get_modality(fields):
    modality = None
    for test_modality in ['dwi', 'flair', 't1']:
        if check_substring(modality_keywords[test_modality], fields):
            # DWI special case: "adc_adc" is an indicator for ADC, not DWI
            if test_modality == 'dwi':
                if any(['adc_adc' in field.lower() for field in fields if field is not None]):
                    continue
            if modality is not None:
                print("*************** WARNING: found conflicting modalities:", modality, test_modality, fields)
                return None
            modality = test_modality
    return modality

def get_modality_from_series(series):
    raise NotImplementedError
    # for modality in ['dwi', 'flair', 't1', 'gre', 'tof', 'adc']:
    #     for identifier in modality_identifiers[modality]:
    #         if identifier in series.lower():
    #             return modality
    # print("Couldn't decode: " + series)
    # return series

def remove_analyze_files(sett):
    return set(filter(lambda f: f[-4:] not in ['.hdr', '.img'], sett))
def find_matching(subj, unknown_modality):
    """
    Matches between the 'original' folders and the 'renamed' ones
    provided to us. There are two ways to match:
    a) make sure the list of files match
    b) use the prefix (each folder starts with 00X_..., and only the stuff
    after the _ was renamed)
    the matchine is done with a) and an assertion is used to check b
    """
    orig_dir = j(DATA_ROOT, subj, unknown_modality)
    filelist = remove_analyze_files(set(os.listdir(orig_dir)))
    renamed = j(RENAMED_ROOT, subj)
    matching_name = None
    for renamed_dir in os.listdir(renamed):
        if renamed_dir.startswith('.'):
            continue
        renamed_filelist = remove_analyze_files(set(os.listdir(j(renamed, renamed_dir))))
        if renamed_filelist == filelist:
            assert matching_name is None
            matching_name = renamed_dir

    if matching_name is not None:
        assert unknown_modality[:3] == matching_name[:3]
        matching_name = matching_name.split('_', 1)[1].lower()
        for unwanted_prefix in ['brain', 'acute_strok']:
            if matching_name.startswith(unwanted_prefix):
                matching_name = matching_name[len(unwanted_prefix):]
    else:
        raise ValueError("couldn't find a folder with that data")

    return matching_name

def get_header_dict(dicom_file, fields=None):
    """
    Returns a dictionary of the header fields in the dicom file.  fields
    specifies which fields to read; if it's None then I'll read all of them
    """
    obj = dicom.read_file(dicom_file, stop_before_pixels=True)
    dictionary = collections.defaultdict(lambda: None)
    if fields is None:
        fields = [k for k in obj.dir() if k != 'PixelData']
    for k in fields:
        val = getattr(obj, k, None)
        if type(val) is dicom.multival.MultiValue:
            val = ' '.join(map(str, val))

        dictionary[k] = val
    return dictionary

def get_modalities(dicomdir):
    """
    Returns a list of tuples; each is
    (dictionary of relevant header stuff, list of ordered scans)
    """
    files = [j(dicomdir, f) for f in sorted(remove_analyze_files(os.listdir(dicomdir)))]
    m = collections.defaultdict(lambda: [])
    fields = ['ProtocolName', 'SeriesDescription', 'ImageType']
    for file in files:
        header = get_header_dict(file, fields)
        key = frozenset(header.iteritems())
        m[key].append(file)

    return [(dict(k), file_list) for (k, file_list) in m.iteritems()]

def find_changing_fields(dicomdir):
    files = [j(dicomdir, f) for f in sorted(remove_analyze_files(os.listdir(dicomdir)))]
    first = files[0]
    base = get_header_dict(first)
    differing_keys = set()
    header_dicts = [base]
    for file in files[1:]:
        current = get_header_dict(file)
        header_dicts.append(current)
        all_keys = set.intersection(set(base.keys()), set(current.keys()))
        for k in all_keys:
            if base[k] != current[k]:
                differing_keys.add(k)
    out = dict(((key, []) for key in differing_keys))
    for header_dict in header_dicts:
        for key in differing_keys:
            out[key].append(header_dict[key])
    return out


def identify_modalities(subjects=None, convert=True):
    ALL_MODALITIES=[]
    fieldnames = ['Subject', 'ProtocolName', 'SeriesDescription', 'ImageType', 'FolderName', 'RenamedFolderName']
    #fieldnames = ['Subject', 'ProtocolName', 'SeriesDescription', 'ImageType', 'FolderName']
    if subjects is None:
        subjects = sorted(os.listdir(DATA_ROOT))
    good_subjects = []
    matrix = []
    try:
        os.remove(DB_FILE)
    except:
        pass
    # db = sqlite3.connect(DB_FILE)
    # db_cursor = db.cursor()
    # column_headings = '(%s)' % ', '.join([ f.lower() + ' text' for f in fieldnames])
    # columns = '(%s)' % ', '.join([ f.lower() for f in fieldnames])

    #db_cursor.execute('CREATE TABLE modalities %s' % column_headings)
    lisa_dict = {}
    with open(LISA_CSV) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        (id_field,) = [f for f in fields if 'id' in f.lower()]
        (flair_field,) = [f for f in fields if 'flair' in f.lower() and 'coronal' not in f.lower()]
        for row in reader:
            subj = row[id_field]
            if subj == '':
                continue
            has_flair = row[flair_field]
            assert has_flair in '01'
            lisa_dict[row[id_field]] = bool(int(has_flair))
    changer_counter = collections.Counter()
    with open(OUT_CSV, 'w') as f, open(RENAMING_CSV, 'w') as ff, open('/dev/null','wb') as null:
        writer = csv.DictWriter(f, ['Subject', 'Folder name', 'Modality'])
        writer.writeheader()

        renaming_writer = csv.DictWriter(ff, fieldnames)
        renaming_writer.writeheader()
        for subj in subjects:
            if not os.path.isdir(j(DATA_ROOT, subj)):
                continue
            # if not subj.endswith('71'):
            #     continue
            subj_in = j(DATA_ROOT, subj)
            subj_out = j(DATA_OUT, subj)
            if not os.path.isdir(subj_out):
                os.mkdir(subj_out)
                os.mkdir(j(subj_out, 'original'))
            subj_modalities = []
            subj_raw_names = []
            subj_unidentified_scans = []
            fail_count = 0
            modality_counter = collections.Counter()
            for unknown_modality in os.listdir(subj_in):
                modality_dir = j(subj_in, unknown_modality)
                if not os.path.isdir(modality_dir):
                    continue
                # possible_sequences = {'Subject': subj}

                # changer_counter.update(find_changing_fields(modality_dir).keys())
                # dicoms = [f for f in os.listdir(modality_dir) if not f.endswith('.hdr') and not f.endswith('.img')]
                # dicoms = [f for f in os.listdir(modality_dir) if f[-4:] not in ['.hdr', '.img']]
                # first_file = sorted(dicoms)[0]

                # #hdr = dicom.read_file(j(subj_in,unknown_modality,'1.dcm'),
                # hdr = dicom.read_file(j(subj_in, unknown_modality, first_file),
                #                       stop_before_pixels=True)

                # ### A few different ways of guessing the modality
                # ## some from the header
                # for field in ['ProtocolName', 'SeriesDescription', 'ImageType']:
                #     attr = getattr(hdr, field, None)
                #     if type(attr) is dicom.multival.MultiValue:
                #         attr = ' '.join(attr)
                #     possible_sequences[field] = attr

                try:
                    modalities = get_modalities(modality_dir)
                except dicom.filereader.InvalidDicomError:
                    continue
                ALL_MODALITIES.extend(modalities)
                if len(modalities) > 1:
                    print("Found %d modalities in one folder." % len(modalities))
                    # if there's more than one modality in the folder, then
                    # the folder name won't help us decode the modalities
                    ignore_folder = True
                else:
                    ignore_folder = False
                for (sequence_clues, file_list) in modalities:
                    possible_sequences = sequence_clues
                    possible_sequences['Subject'] = subj
                    ## the folder name
                    if not ignore_folder:
                        try:
                            fn = unknown_modality.split('_', 1)[1].lower()
                        except IndexError:
                            continue
                        for unwanted_prefix in ['brain', 'acute_strok']:
                            if fn.startswith(unwanted_prefix):
                                fn = fn[len(unwanted_prefix):]
                        possible_sequences['FolderName'] = fn
                        # TODO uncomment this since MGH renamed things
                        possible_sequences['RenamedFolderName'] = find_matching(subj, unknown_modality)
                    else:
                        possible_sequences['FolderName'] = possible_sequences['RenamedFolderName'] = ''
                        #possible_sequences['FolderName'] = ''
                    renaming_writer.writerow(possible_sequences)
                    # db_cursor.execute('INSERT INTO modalities %s VALUES (' % columns + ', '.join(['?'] * len(fieldnames)) + ');',
                    #         [possible_sequences[field] for field in fieldnames])

                    suspected_modality = get_modality([possible_sequences[clue] for clue in ['ProtocolName', 'SeriesDescription', 'FolderName', 'RenamedFolderName']])
                    #suspected_modality = get_modality([possible_sequences[clue] for clue in ['ProtocolName', 'SeriesDescription', 'FolderName']])
                    if suspected_modality is None:
                        subj_unidentified_scans.append((modality_dir, sequence_clues))
                        continue
                    suspected_modality = suspected_modality.replace(' ','')



                        #print("**** Couldn't figure out folder {0}/{1}".format(subj, unknown_modality))
                    #suspected_modality = modality_mappings[sequence]
                    # subj_raw_names.append(sequence)
                    # suspected_modality = get_modality_from_series(sequence)
                    subj_modalities.append(suspected_modality)

                    scandir_out = j(subj_out,
                                    'original',
                                    '%s_%d' % (suspected_modality, modality_counter[suspected_modality]+1))

                    try:
                        os.mkdir(scandir_out)
                    except:
                        pass
                    out_file_name = j(scandir_out, '%s_%s_raw.nii.gz' % (subj, suspected_modality))
                    success = True
                    if convert:
                        try:
                            os.mkdir(scandir_out)
                        except:
                            pass
                        print("Converting {} in {}".format(suspected_modality, modality_dir))
                        tmpdir = tempfile.mkdtemp()
                        for file in file_list:
                            os.symlink(file, j(tmpdir, os.path.basename(file)))

                        ret = subprocess.call(['mri_convert',
                                        j(tmpdir, os.path.basename(file_list[0])),
                                        out_file_name],
                                       stdout=null)# mri_convert is noisy, so redirect stdout to /dev/null
                        if ret != 0:
                            print("************************* mri_convert failure")
                            continue

                        if suspected_modality == 'dwi' or suspected_modality == 'flair':
                            had_multiple_volumes = collapse_volumes(out_file_name)
                            if suspected_modality == 'dwi' and not had_multiple_volumes:
                                print("************************* failure for subj %s DWI: %s" % (subj, unknown_modality))
                                shutil.rmtree(scandir_out)
                                success = False
                        shutil.rmtree(tmpdir)
                    if success:
                        modality_counter[suspected_modality] += 1
                    out = {'Subject': subj, 'Folder name': unknown_modality,
                            'Modality': suspected_modality}
                    writer.writerow(out)
            matrix.append(['t1' in subj_modalities, 'flair' in subj_modalities, 'dwi' in subj_modalities])
            subj_raw_names.extend(['?'] * fail_count)
            if 'flair' not in subj_modalities and lisa_dict[subj]:
                print("~" * 70)
                print("I couldn't find a FLAIR for subject %s even though Lisa says there is one. Here's what I found but couldn't ID:" % subj)
                for (loc, clues) in subj_unidentified_scans:
                    print(loc)
                    print(clues)
                    print()
            if 'flair' in subj_modalities and 'dwi' in subj_modalities:
                good_subjects.append(subj)
            else:
                duplicate_modalities = len(subj_modalities) - len(set(subj_modalities))
                if duplicate_modalities > 0:
                    print("I found %d duplicate modalities for %s" % (duplicate_modalities, subj))
                    print(sorted(subj_modalities))
                # print(subj)
                # print(subj_modalities)
                # print(subj_raw_names)
                # print()
                pass
    # print(good_subjects)
    # print(len(good_subjects))
    #print(matrix)
    #db.commit()
    print(matrix)
    with open(BINARY_CSV, 'w') as bcsv:
        writer = csv.writer(bcsv)
        writer.writerow(["Subject", "Has T1", "Has FLAIR", "Has DWI"])
        for (i, subj) in enumerate(subjects):
            writer.writerow([subj] + map(int, matrix[i]))

    return (matrix, changer_counter)
    # return ALL_MODALITIES

if __name__ == '__main__':
    site_number = sys.argv[1]
    site_name = 'site' + site_number
    process_date =  sys.argv[2]
    os.environ['PATH'] += ':/data/vision/polina/shared_software/freesurfer_v5.1.0/bin/'
    global STROKE
    STROKE='/data/vision/polina/projects/stroke'
    global DATA_ROOT
    DATA_ROOT= j(STROKE, 'raw_datasets', site_number)
    global RENAMED_ROOT
    RENAMED_ROOT= j(STROKE, 'received_work', site_name, 'renamed')
    global DATA_OUT
    DATA_OUT= j(STROKE, 'processed_datasets', process_date, site_name)
    global OUT_CSV
    OUT_CSV= j(STROKE, 'work', 'rameshvs', site_name + '.csv')
    global DB_FILE
    DB_FILE= j(STROKE, 'work', 'rameshvs', site_name + '.db')
    global LISA_CSV
    LISA_CSV = j(STROKE, 'received_work', 'site_modality_info_csv', site_name + '.csv')
    global RENAMING_CSV
    RENAMING_CSV= j(STROKE, 'work', 'rameshvs', site_name + '_renaming.csv')
    global BINARY_CSV
    BINARY_CSV= j(STROKE, 'work', 'rameshvs', site_name + '_binary.csv')
    (matrix, changer_counter) = identify_modalities(convert=True)

