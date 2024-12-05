# neuriteX
A collection of functions to evaluate the integrity of neuronal processes (neurites) in microscopy images

## Description
For neurons cultured in a dish, a score is computed that reflects the integrity of neuronal processes, or neurites (i.e. axons and dendrites). The code is primarily intended for neurodegeneration research in academia and industry. The codebase is written in Python (version 3.6.7, 3.7.9) and R (version 4.4.1) and has been tested in Windows 10.

Functionalities implemented in Python include: image correction; neurite segmentation with computation of raw score, corrected score, and neurite integrity index (NII); and generation of images representing processing intermediates (for optimization of parameters).  
Charts for visualization of multiple conditions are created in R.

The provided step-by-step guide requires knowledge of Python and R. The use of IDEs (e.g. PyCharm and R Studio) is recommended.  ImageJ (or Fiji) is not an absolute requirement, but is helpful to look at some of the generated output images.


## Analysis – first steps

Perform the following steps:

-	Download repository from https://github.com/pmeran/neuriteX/ and extract locally
-	In PyCharm, create new root folder named 'neuriteX_root_folder/'
-	From downloaded repository, transfer folder `img_ori/` and file `neuriteX.py` to root folder
-	Open `neuriteX.py` in PyCharm
-	Run `neuriteX.py` code from first line all the way down to `END OF METHODS`
-	Move to section `# S_1 - single image analysis` and find the following line of code:<br />
  `path_main = '<path_main>/neuriteX_root_folder/'`<br />
  then replace `<path_main>` with absolute path for root folder<br />

## S_1 – single image analysis

This section includes the following operations:<br />
- **S_1 - read image**<br />
Read and display image from folder `img_ori/`<br />
- **S_1 - image correction**<br />
Noise reduction, brightness adjustment, gamma correction<br />

Function:<br />
`imgC, status = nX_correction(imgX, promMin = 5, perc_hi = 95, lim_I_new = 200, gamma = 0.8, win = 3, ord = 2)`<br />
<br />
Returns:<br />
`imgC`	corrected output image<br />
`status`	returns 1 if correct, or error code -1 if input image has no detectable structures<br />
<br />
Parameters:<br />
`imgX`	input image (numpy array, 1196 x 1196, np.uint8)<br />
`promMin`	minimum peak prominence<br />
`perc_hi`	percentile of detected peak intensities to use as input intensity pivot for brightness adjustment<br />
`lim_I_new`	output intensity pivot for brightness adjustment<br />
`gamma`	gamma correction parameter <br />
`win`	window size for Savitzky-Golay noise reduction<br />
`ord`	order for Savitzky-Golay noise reduction<br />
<br />
- **S_1 - image segmentation, generation of test images**<br />
Generate raw neurite integrity score, create stack of images showing intermediate processing states<br />

Function:<br />
`D, stack = nX_segmentation_test (imgC, img_file, ptUL=(10,10), eH=100, extF = 3, win = 3, ord=2, t = 100000)`<br />
<br />
Returns:<br />
`D`	dictionary with output `N_perc`, the percentage of neurite pixels among all peak pixels<br />
`stack`	stack of images representing various stages of analysis (for parameter adjustments)<br />
<br />
Parameters:<br />
`imgC`	(corrected) input image (numpy array, 1196 x 1196, np.uint8)<br />
`img_file`	filename of original image<br />
`ptUL`	upper left anchor of image area to be analyzed<br />
`eH`	half edge length of image area to be analyzed<br />
`extF`	scaling factor for analysis<br />
`win`	window size for Savitzky-Golay noise reduction<br />
`ord`	order for Savitzky-Golay noise reduction<br />
`t`	number of peak pixels to be selected randomly for image analysis<br />
<br />
- **S_1 – image segmentation**<br />
Generate raw neurite integrity score<br />

Function:<br />
`D = nX_segmentation (imgC, img_file, ptUL=(10,10), eH=100, extF=3, win=3, ord=2, t=100000)`<br />
<br />
Function `nX_segmentation` does not return `stack` but is otherwise identical to function `nX_segmentation_test`<br />


          






