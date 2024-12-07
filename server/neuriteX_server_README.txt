# © 2024 Paul Meraner
# neuriteX

neuriteX server scripts were run successfully on a Linux compute cluster (Red Hat Enterprise Linux version 9.4) with slurm scheduler, running Python version 3.6.7.

Instructions

- Create project folder in Linux environment (e.g. 'neuriteX_project_folder/')
- From neuriteX repository, transfer folder 'img_ori' to Linux project folder
- From neuriteX repository, transfer the following files to Linux project folder:
	correction.sh
	segmentation.sh
	neuriteX_methods.py
	neuriteX_correction.py
	neuriteX_segmentation.py
- Schedule image correction job using slurm:
	sbatch correction.sh
  Original .png images in 'img_ori' will be corrected and stored as _corr.png files in folder 'img_corr'. Error codes are listed in output file 'df_corr.pkl' (pandas dataframe).
- Schedule image segmentation job using slurm:
	sbatch segmentation.sh
  Image segmentation is slow (3-6 hours for ~1000 images).  Neurite integrity scores are stored in output file 'df_seg.pkl' (pandas dataframe).
