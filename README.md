# stroke_processing - processing FLAIR images for stroke study

This repository contains code for registration of FLAIR images and segmentation
of white matter hyperintensity.


## Getting started

1. First, download [pipebuilder](https://github.com/rameshvs/pipebuilder)
   and [pyniftitools](https://github.com/rameshvs/pyniftitools). In your
   pipebuilder config file (`site.cfg`), specify the location where you downloaded
   `pyniftitools`.

2. Set the appropriate paths in `stroke.cfg`.
   * `site_subject_lists` is a folder with per-site lists of all the subjects
     (`site04.txt`, `site16.txt`, etc).
   * `processing_root` is where you want to save the results to. See also `processed_0204`
   * `atlas_base` is the location of the atlas files
   * `processed_0204` is currently a hardcoded version of `processing_root`, so if you change `processing_root` you should change this as well.
   * `site_average_path` is a location to save average maps for each site to
   * `volume_csv_path` is a location to save CSV files with WMH volumes to

3. Run the registration using `scripts/process_stroke_site`. For example, to
   process site 16:

        $ scripts/process_stroke_site.sh 16

   The pipeline code for this is located in
   `stroke_processing/registration/flairpipe.py`. Each Command has a short
   description of what it does.
