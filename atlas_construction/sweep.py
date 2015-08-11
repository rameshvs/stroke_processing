#!/usr/bin/env python
from __future__ import print_function
import pipebuild as pb
import os

atlas = pb.Atlas('buckner62', '/data/vision/polina/projects/stroke/work/input/atlases/')
data_subfolder = '2013_12_13/site00'
DATA_ROOT = os.path.join(pb.NFS_ROOT, 'projects/stroke')
#BASE = os.path.join(DATA_ROOT, 'processed_datasets', data_subfolder)
BASE = os.path.join('/data/vision/scratch/polina/users/rameshvs', data_subfolder)
dataset = pb.Dataset(
            BASE,
            atlas,
            # How are the inputs to the pipeline stored?
            os.path.join(BASE , '{subj}/original/{modality}_1/{subj}_{modality}_{feature}'),
            # How should intermediate files be stored?
            os.path.join(BASE, '{subj}/images/{subj}_{modality}_{feature}{modifiers}'))

# lists = '/data/vision/polina/projects/stroke/work/misc_lists/'
# # r1s = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
# # r2s = [0.0625, 0.0883883476483, 0.125, 0.176776695297, 0.25, 0.353553390593, 0.5, 0.707106781187, 1.0]
# SUBJECTS=[10553, 10583, 10613, 10807, 10958, 11394, 11410, 11453, 11470]

r1s = [float(x) for x in open(lists + 'registration_parameters_tweaked.txt')]
r2s = [float(x) for x in open(lists + 'registration_parameters_small_tweaked.txt')]
# for (i,regularization1) in enumerate(r1s):
#     for (j,regularization2) in enumerate(r2s):
#         pngs = []
#         for (k,subj) in enumerate(SUBJECTS):
#             nname = ('%0.3f_%0.3f' % (regularization1, regularization2)).replace('.','')
#             #filename = dataset.get(subj, 'other', 'sweep_overlay_horiz_%s'%nname, extension='.png')
#             filename = dataset.get(subj, 'other', 'sweep_overlay_horiz_sixup_%s'%nname, extension='.png')
#             cmd = ' '.join(['cp', filename, '/afs/csail.mit.edu/u/r/rameshvs/public_html/tipix/interrater_sweep_bysubj_sixups_bigger_brainreg/%d_%d_%d.png' % (k+1, i+1, j+1)])
#             print(cmd)
#             # pngs.append(filename)
#         # cmd = ' '.join(['convert'] + pngs + ['+append', '/data/vision/polina/projects/stroke/work/interrater_sweep/%d_%d.png' % (i+1, j+1)])
#         # print(cmd)

