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
moving = atlas.get_original(feature='flairTemplateInBuckner_sigma8')
seg = atlas.get_original(feature='buckner61_seg')
fixed = data.get_original(feature='template')
mask = data.get_original(feature='mask')
print(moving)
print(seg)
print(fixed)
print(mask)
for (i,r1) in enumerate(r1s):
    for (j,r2) in enumerate(r2s):

            masked_reg = pb.ANTSCommand(
                    'Registration with mask',
                    moving=moving,
                    fixed=fixed,
                    regularization='Gauss[%0.3f,%0.3f]' % (r1, r2),
                    metric='CC',
                    radiusBins=32,
                    output_folder=data.get_folder(feature=''),
                    method='200x200x200',
                    mask=mask)

            warp = pb.ANTSWarpCommand.make_from_registration(
                        'Warp segmentation using registration',
                        moving=seg,
                        reference=fixed,
                        registration=masked_reg,
                        output_folder=data.get_folder(feature=''),
                        inversion='forward',
                        useNN=True,
                        )
            warped_labels = warp.outfiles[0]
            filename = data.get(feature='seg_{}_{}'.format(i, j))
            filename_png = data.get(feature='tipix/seg_{}_{}'.format(i, j), extension='.png')
            #filename_png = '/data/vision/polina/projects/stroke/work/rameshvs/site_atlases/site07/atlas_atlas_reg/seg_{}_{}.png'.format(i, j) 
            pb.InputOutputShellCommand(
                    'rename',
                    cmdName='cp',
                    input=warped_labels,
                    output=filename)
            # tools.better_overlay(
            #         seg_file=filename,
            #         flair_img=fixed,
            #         slices=[127, 147, 154, 161],
            #         out_filename=filename_png,
            #         )

            pb.Command.generate_code_from_datasets([data,atlas], data.get_log_folder(), short_id='%d_%d'%(i,j),
                    sge=True,wait_time=1)
            pb.Command.clear()
