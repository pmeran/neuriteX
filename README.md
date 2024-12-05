# neuriteX
A collection of functions to evaluate the integrity of neuronal processes (neurites) in microscopy images

## Description
For neurons cultured in a dish, a score is computed that reflects the integrity of neuronal processes, or neurites (i.e. axons and dendrites). The code is primarily intended for neurodegeneration research in academia and industry. The codebase is written in Python (version 3.6.7, 3.7.9) and R (version 4.4.1) and has been tested in Windows 10.

Functionalities implemented in Python include: image correction; neurite segmentation with computation of raw score, corrected score, and neurite integrity index (NII); and generation of images representing processing intermediates (for optimization of parameters).

Charts for visualization of multiple conditions are created in R.

The provided step-by-step guide requires knowledge of Python and R, and the use of IDEs (e.g. PyCharm and R Studio).  ImageJ (or Fiji) is helpful to look at some of the generated outputs.


## Analysis – first steps

Perform the following steps

-	Download repository from https://github.com/pmeran/neuriteX/ and extract locally
-	In PyCharm, create new root folder named 'neuriteX_root_folder/'
-	From downloaded repository, transfer folder 'img_ori/' and file 'neuriteX.py' to root folder
-	Open `neuriteX.py` in PyCharm
-	Run `neuriteX.py` code from first line all the way down to `END OF METHODS`
-	Move to section `# S_1 - single image analysis` and find the following line of code:
  `path_main = '<path_main>/neuriteX_root_folder/'`
  then replace `<path_main>` with absolute path for root folder

Section `# S_1 – single image analysis`

`# S_1 - read image`
`# S_1 - image correction`
`# S_1 - image segmentation test`
`# S_1 – image segmentation`






