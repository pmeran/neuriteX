# neuriteX
A collection of functions to evaluate the integrity of nerve cell processes (neurites) in microscopy images

## Description
For neurons (nerve cells) expressing a fluorescent cytoplasmic marker, the integrity of neurites (axons and dendrites) is quantitatively evaluated, and a score is returned. The code is primarily intended for research laboratories in academia and industry that study neurodegeneration. The codebase is written in Python (version 3.6.7, 3.7.9) and R (version 4.4.1) and has been tested in Windows 10.

Image analysis is implemented in Python and includes: image correction; neurite segmentation with computation of the neurite integrity index (NII); and generation of images representing processing intermediates (for parameter optimization).<br />
Result visualization is implemented in R.<br />
<br />
A step-by-step guide is provided that requires knowledge of Python and R, as well as familiarity with the use of IDEs (PyCharm and R Studio).  ImageJ (or Fiji) is helpful for evaluation of generated image stacks.<br />


## Analysis – first steps

*	Download repository from https://github.com/pmeran/neuriteX/
*	Open PyCharm, create new project folder 'neuriteX_root_folder/'
*	From downloaded repository, transfer folder `img_ori/` and file `neuriteX.py` to created root folder in PyCharm
*	Open `neuriteX.py` in PyCharm and do the following:<br />
&nbsp;&nbsp;o&nbsp;&nbsp;&nbsp;To import modules and load methods, run code in `neuriteX.py` from first line all the way down to `END OF METHODS`<br />
&nbsp;&nbsp;o&nbsp;&nbsp;&nbsp;Move to section `# 1. Single image analysis` and find the following line of code:<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`path_main = '<path_main>/neuriteX_root_folder/'`<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;then replace `<path_main>` with the absolute path for `neuriteX_root_folder/`<br />

## 1. Single image analysis

**Summary**<br />

To get familiar with the analysis pipeline, it is recommended to first analyze single original images (provided in folder `img_ori`) by running code section `# 1. Single image analysis` in `neuriteX.py`.<br /><br />
Images are analyzed in two sequential steps, using two segmentation filters. The first filter identifies pixels corresponding to peaks in cross-sectional intensity profiles. The second filter examines the local surroundings of pixels identified in the first step, and selects them if they are part of neurites (curvilinear structures) or rejects them if they are part of blebs (small spherical or elliptical structures).<br />
For each image, a raw numerical score `N_perc` (for neurite percent) is calculated, defined as the percentage of pixels passing the 2nd filter (neurites) versus pixels passing the 1st filter (peaks). `N_perc` can be retrieved from variable `D` as `D['N_perc']`, which is returned by functions `nX_segmentation_test()` and `nX_segmentation()` (further explained below).<br /><br />

**Fig. 1.a&nbsp;&nbsp;Neurite segmentation**<br />
Images of intact neurites (top left) and degenerated neurites (bottom left) were analyzed using two sequential segmentation filters.<br />
A raw score for neurite integrity is calculated as the percentage of pixels passing the 2nd filter versus pixels passing the 1st filter.<br />
Image size 400 x 400 pixels, or 88.4 x 88.4 µm.<br />
<img src="neuriteX_Fig.1a.PNG" width="600"/>
<br /><br />
**Fig. 1.b&nbsp;&nbsp;Neurite segmentation using simulated images.**<br /><br />
<img src="neuriteX_Fig.1b.PNG" width="600"/>
<br /><br />

Details for modules in section `#1. Single image analysis` in `neuriteX.py`:<br />
- **1.1&nbsp;&nbsp;Display image**<br />
Read and display image from folder `img_ori/`<br /><br />
- **1.2&nbsp;&nbsp;Image correction**<br />
Reduce noise, adjust brightness, add gamma correction<br /><br />
Function:<br />
`imgC, status = nX_correction(imgX, promMin = 5, perc_hi = 95, lim_I_new = 200, gamma = 0.8, win = 3, ord = 2)`<br /><br />
Returns:<br />
`imgC`	corrected output image<br />
`status`	returns 1 if correct, or error code -1 if input image has no detectable structures<br /><br />
Parameters:<br />
`imgX`	input image (numpy array, 1196 x 1196, np.uint8)<br />
`promMin`	minimum peak prominence<br />
`perc_hi`	percentile of detected peak intensities to use as input intensity pivot for brightness adjustment<br />
`lim_I_new`	output intensity pivot for brightness adjustment<br />
`gamma`	gamma correction parameter <br />
`win`	window size for Savitzky-Golay noise reduction<br />
`ord`	order for Savitzky-Golay noise reduction<br /><br />
- **1.3&nbsp;&nbsp;Image segmentation and generation of test images**<br />
Generate neurite integrity score, create images of intermediate processing states<br /><br />
Function:<br />
`D, stack = nX_segmentation_test (imgC, img_file, ptUL=(10,10), eH=100, extF = 3, win = 3, ord=2, t = 100000)`<br /><br />
Returns:<br />
`D`	dictionary with output `N_perc`, the percentage of neurite pixels among all peak pixels<br />
`stack`	stack of images representing various stages of analysis (for parameter adjustments)<br /><br />
Parameters:<br />
`imgC`	(corrected) input image (numpy array, 1196 x 1196, np.uint8)<br />
`img_file`	filename of original image<br />
`ptUL`	upper left anchor of image area to be analyzed<br />
`eH`	half edge length of image area to be analyzed<br />
`extF`	scaling factor for analysis<br />
`win`	window size for Savitzky-Golay noise reduction<br />
`ord`	order for Savitzky-Golay noise reduction<br />
`t`	number of peak pixels to be selected randomly for image analysis<br /><br />
- **1.4&nbsp;&nbsp;Image segmentation**<br />
Generate neurite integrity score<br /><br />
Function:<br />
`D = nX_segmentation (imgC, img_file, ptUL=(10,10), eH=100, extF=3, win=3, ord=2, t=100000)`<br /><br />
Function `nX_segmentation()` is largely identical to function `nX_segmentation_test()`, with the difference that the former does not return `stack` (a stack of images representing processing intermediates). `nX_segmentation()` is faster and is used for batch processing.<br />
<br /><br />

## 2. Batch processing of multiple images

**Summary**<br />
Neurodegeneration experiments require comparison of multiple conditions set up in multiplicates. We therefore implemented processing of images in batch mode, and provide both code and sample images for a batch analysis test run.<br />
Folder `img_ori` contains sample images from a neurodegeneration experiment involving axotomy. The provided images represent two conditions ('uncut', 'cut'), 3 scenes (i.e. imaging areas) per condition, and 7 acquisition time points for each scene, resulting in a total of 2 * 3 * 7 = 42 images. Spreadsheet `df_excel.xlsx` lists all images and their relation to scenes, conditions, and time points. Since the spreadsheet can get quite complex for larger experiments, and is an integral part for creating charts, it should be generated programmatically, as shown (code section `Create excel spreadsheet` in file `neuriteX.py`).<br /><br />
Code for batch image processing is structured into the following two sections, which should be run sequentially:<br />
- **2.1&nbsp;&nbsp;Batch processing - image correction**<br />
- **2.2&nbsp;&nbsp;Batch processing - image segmentation**

Both modules make use of the same functions used for single image processing.<br /><br />
Image segmentation is very time intensive, posing challenges for parameter optimization and resulting in very long run times.<br />
To facilitate <ins>parameter optimization</ins>, both segmentation functions `nX_segmentation()` and `nX_segmentation_test()` offer the option to minimize the analyzed image area (by tweaking parameters `pUL` = upper left corner, and `eH` = half edge of image area to be analyzed). As a result, setting `eH = 100` takes about 15 seconds; while `eH = 580` takes about 6 minutes to complete.<br />
<ins>Run times</ins> were considerably shortened by parallelization. To cope with run times of several hours, we successfully ran the code without major tweaks on a Linux compute cluster (sample scripts are given in folder `src_cluster`).<br /><br />
Batch processing generates a raw neurite integrity score `N_perc` for each image, and stores values in file `df_seg.pkl`.<br /><br />

## 3. Scoring neurite integrity
This section merges experiment information in df_excel.xlsx with neurite integrity scores: N_perc (df_R1.csv), N_perc_corr (df_R2.csv), and NII (file df_R3.csv).<br />
- **N_perc**<br />
N_perc is a raw score of neurite integrity, defined as the percentage of neurite pixels (2nd filter) versus cross-sectional peaks (1st filter) (Fig. 1a, 2). N_perc normalizes for area and overall image complexity.<br />
- **N_perc_corr**<br />
N_perc_corr results from correction of N_perc, which to some extent is affected by background variability.<br />
- **NII**<br />
The neurite integrity index (NII) results from normalization of N_perc_corr, using the earliest time point (4 h or earlier) of N_perc_corr as reference.<br />
<br />
Charts for all three scores are generated with neuriteX.R, taking files df_R1.csv, df_R2.csv, and df_R3.csv as input.<br />
<br />
<img src="neuriteX_Fig.3.PNG" width="800"/>













          






