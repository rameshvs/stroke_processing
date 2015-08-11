import numpy as np
import nibabel as nib

def process_sites(sites=(3, 4, 7, 16, 18, 20, 21)):
    sites_and_maps = load_wmh_maps(sites)
    process_loaded_sites(sites_and_maps)

def process_loaded_sites(sites_and_maps):
    out = []
    for (site, map) in sites_and_maps:
        (u, s, v) = compute_pca(map)
        save_pca(v, site)
        out.append((site, map, (u, s, v)))
    return out

def save_pca(v, site, n_components=10):
    atlas_seg = nib.load('/data/vision/polina/projects/stroke/work/input/atlases/buckner61_seg.nii.gz').get_data()
    wm_mask = np.logical_or(atlas_seg == 2, atlas_seg == 41)
    atlas_nii = nib.load('/data/vision/polina/projects/stroke/work/input/atlases/buckner61.nii.gz')
    for i in xrange(n_components):
        pca = np.zeros(atlas_nii.shape)
        pca[wm_mask] = v[i, :]
        out = nib.Nifti1Image(pca, header=atlas_nii.get_header(), affine=atlas_nii.get_affine())
        out.to_filename('/data/vision/polina/projects/stroke/work/rameshvs/sites/pca/site{site:02d}/{i}.nii.gz'.format(site=site, i=i))

def load_wmh_maps(sites=(3, 4, 7, 16, 18, 20, 21)):
    subj_list_template = '/data/vision/polina/projects/stroke/work/subject_lists/sites/site%02d.txt'
    atlas_seg = nib.load('/data/vision/polina/projects/stroke/work/input/atlases/buckner61_seg.nii.gz').get_data()
    wm_mask = np.logical_or(atlas_seg == 2, atlas_seg == 41)
    wmh_maps = []
    for site in sites:
        print("Begin loading site %d" % site)
        subject_file = subj_list_template % site
        with open(subject_file) as f:
            subjects = f.read().split()
        wmh = np.zeros([wm_mask.sum(), len(subjects)])
        wmh = []
        i=0
        good_subjects = []
        for (i, subj) in enumerate(subjects):
            try:
                nii = nib.load('/data/vision/polina/projects/stroke/processed_datasets/2015_02_04/site{site:02d}/{subj}/images/{subj}_wmh_threshold_seg_in_atlas.nii.gz'.format(site=site, subj=subj))
            except IOError:
                continue
            masked = nii.get_data()[wm_mask]
            wmh.append(masked)
            good_subjects.append(subj)
            i += 1

        print("Done loading site %d" % site)
        wmh_maps.append(np.vstack(wmh))
    return zip(sites, wmh_maps)

def compute_pca(wmh_map):
    """ wmh_map is N_subj x N_vox """
    average_image = wmh_map.mean(0)
    (u, s, v) = np.linalg.svd(wmh_map - average_image, full_matrices=False)
    return (u, s, v)
