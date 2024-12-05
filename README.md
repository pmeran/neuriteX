# neuriteX
A collection of functions to evaluate the integrity of neuronal processes (neurites) in microscopy images

## Description
For neurons modified with a fluorescent cytoplasmic marker, a score is computed that reflects the integrity of all neurites (axons and dendrites). The code is primarily intended for research labs in academia and industry that study neurodegeneration. The codebase is written in Python (version 3.6.7, 3.7.9) and R (version 4.4.1) and has been tested in Windows 10.

Image analysis is implemented in Python and includes: image correction; neurite segmentation with computation of raw score, corrected score, and neurite integrity index (NII); and generation of images representing processing intermediates (for optimization of parameters).<br />
Result visualization is implemented in R.<br />
<br />
The provided step-by-step guide requires knowledge of Python and R and the use of IDEs (PyCharm and R Studio).  ImageJ (or Fiji) is not required but is helpful to look at generated image stacks.<br />


## Analysis – first steps

-	Download repository from https://github.com/pmeran/neuriteX/ and extract locally
-	In PyCharm, create new root folder named 'neuriteX_root_folder/'
-	From downloaded repository, transfer folder `img_ori/` and file `neuriteX.py` to root folder
-	Open `neuriteX.py` in PyCharm
-	Run `neuriteX.py` code from first line all the way down to `END OF METHODS`
-	Move to section `# S_1 - single image analysis` and find the following line of code:<br />
  `path_main = '<path_main>/neuriteX_root_folder/'`<br />
  then replace `<path_main>` with absolute path for root folder<br />

## 1. Single image analysis

Summary<br />

To get familiar with the analysis pipeline, it is recommended to first analyze single original images (provided in folder `img_ori`) by running code section `# 1. Single image analysis` in `neuriteX.py`.<br /><br />
In brief, after intensity correction, images are analyzed with two sequential segmentation filters. The first filter identifies pixels corresponding to peaks in cross-sectional intensity profiles. These pixels pass the second filter if - based on examination of their local surroundings - they represent neurites (curvilinear structures) but not blebs (small spherical or elliptical structures).<br />
A raw numerical score `N_perc` (for neurite percent) is calculated as the percentage of pixels passing the 2nd filter (neurites) versus pixels passing the 1st filter (peaks). `N_perc` can be retrieved from output dictionary `D` as `D['N_perc']`.<br />

Examples of segmentation performance with real images.<br />

<img src="demo_image_2.PNG" width="800"/>
<br /><br /><br />

- **1.1&nbsp;&nbsp;&nbsp;Read and display image**<br />
Read and display image from folder `img_ori/`<br /><br />
- **1.2 Image correction**<br />
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
- **1.3 Image segmentation and generation of test images**<br />
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
- **1.4 Image segmentation**<br />
Generate neurite integrity score<br /><br />
Function:<br />
`D = nX_segmentation (imgC, img_file, ptUL=(10,10), eH=100, extF=3, win=3, ord=2, t=100000)`<br /><br />
Function `nX_segmentation` is largely identical to function `nX_segmentation_test`, with the difference that the former does not return `stack`. `nX_segmentation` is therefore faster and is used for batch processing.<br />
<br /><br />







          






