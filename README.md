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

Function:
`imgC, status = nX_correction(imgX, promMin = 5, perc_hi = 95, lim_I_new = 200, gamma = 0.8, win = 3, ord = 2)`

`imgC`	output image
`status`	returns 1 if correct, -1 if problematic (e.g. input image is black, has no detectable structures

`imgX`	input image (numpy array, 1196 x 1196, np.uint8)
`promMin`	minimum peak prominence
`perc_hi`	percentile of detected peak intensities to use as input pivot for brightness adjustment
`lim_I_new`	output pivot intensity for brightness adjustment
`gamma`	parameter for gamma correction
`win`	window size for Savitzky-Golay noise reduction
`ord`	order for Savitzky-Golay noise reduction








<br />
- **S_1 - image segmentation and test images**<br />
Generate raw neurite integrity score, create stack of images showing intermediate processing states<br /> 
- **S_1 – image segmentation**<br />
Generate raw neurite integrity score<br />

          






