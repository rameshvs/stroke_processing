import pipebuild as pb
import tools
import sys

site = sys.argv[1]
base_dir = '/data/vision/polina/projects/stroke/work/rameshvs/site_atlases/site%s' % site
data = pb.Dataset(
        base_dir=base_dir,
        original_template='site%s_template_{feature}' % site,
        processing_template='atlas_atlas_reg/{feature}',
        log_template='logs',
        )

atlas_dir = '/data/vision/polina/projects/stroke/work/input/atlases/'
atlas = pb.Dataset(
        base_dir=atlas_dir,
        original_template='{feature}',
        processing_template='{feature}',
        log_template='logs',
        )

r1s = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
r2s = [0.0, 0.0625, 0.0883883476483, 0.125, 0.176776695297, 0.25, 0.353553390593, 0.5, 0.707106781187, 1.0, 2.0, 3.0]
fixed = data.get_original(feature='template')

for (i,r1) in enumerate(r1s):
    for (j,r2) in enumerate(r2s):
            filename = data.get(feature='seg_{}_{}'.format(i, j))
            filename_png = data.get(feature='tipix/seg_{}_{}'.format(i, j), extension='.png')
            try:

                tools.better_overlay(
                        seg_file=filename,
                        flair_img=fixed,
                        slices=[127, 147, 154, 161],
                        out_filename=filename_png,
                        )
            except IOError:
                print("failure on (%d,%d)" % (i,j))
