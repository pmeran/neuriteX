

# © 2024 Paul Meraner
# neuriteX

neuriteX server scripts were run successfully on a Linux compute cluster (Red Hat Enterprise Linux version 9.4) with slurm scheduler, running Python version 3.6.7.

Instructions

- Create project folder in Linux environment (e.g. 'neuriteX_project_folder/')
- From neuriteX repository, transfer folder 'img_ori' to project folder
- From neuriteX repository, transfer the following files to project folder:
	correction.sh
	segmentation.sh
	neuriteX_methods.py
	neuriteX_correction.py
	neuriteX_segmentation.py
- Schedule image correction job using slurm:
	sbatch correction.sh
  This will create a new folder 'img_corr' containing corrected images, and output file 'df_corr.pkl' with error codes (a pandas dataframe).
- Schedule image segmentation job using slurm:
	sbatch segmentation.sh
  Image segmentation is slow (ca. 3-6 hours for 1000+ images).
  Neurite integrity scores are stored in output file 'df_seg.pkl' (a pandas dataframe).


	


-rwxr-x---. 1 meraner FreemanLab   109 Dec  7 14:13 correction.sh
-rw-r-----. 1 meraner FreemanLab  1703 Dec  7 14:14 df_corr.pkl
-rw-r-----. 1 meraner FreemanLab  8218 Dec  7 14:17 df_seg.pkl
drwxr-s---. 2 meraner FreemanLab  4096 Dec  7 14:14 img_corr
drwxr-s---. 2 meraner FreemanLab  4096 Dec  7 14:11 img_ori
-rwxr-x---. 1 meraner FreemanLab  1087 Dec  7 14:10 neuriteX_correction.py
-rwxr-x---. 1 meraner FreemanLab 31731 Dec  7 14:10 neuriteX_methods.py
-rwxr-x---. 1 meraner FreemanLab   845 Dec  7 14:10 neuriteX_segmentation.py
-rwxr-x---. 1 meraner FreemanLab   115 Dec  7 14:14 segmentation.sh
