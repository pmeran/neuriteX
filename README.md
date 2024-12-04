# neuriteX
A collection of functions to evaluate the integrity of neuronal processes (neurites) in microscopy images

## Description
For neurons cultured in a dish, a score is computed that reflects the integrity of neuronal processes, or neurites (i.e. axons and dendrites). The code is primarily intended for neurodegeneration research in academia and industry. The codebase is written in Python (version 3.7.9) and R (version 4.4.1) and has been tested in Windows 10.

Functionalities implemented in Python include: image correction; neurite segmentation with computation of raw score, corrected score, and neurite integrity index (NII); and generation of images representing processing intermediates (for optimization of parameters).

Charts for visualization of multiple conditions are created in R.

The provided step-by-step guide requires knowledge of Python and R, and the use of IDEs (e.g. PyCharm and R Studio).

## Images
For code development, 8-bit grayscale images in .png format were used, with a size of 1196 x 1196 pixels and a pixel side length corresponding to 0.663 um (mouse axons typically have a diameter of 0.6-1.1 um, but can form larger bundles). Other image formats and sizes may be used but may require modification of internal parameters (e.g. average axon thickness in pixels). The provided code also produces meaningful results for inverted phase contrast images, which have traditionally been used in neurodegeneration research. However, fluorescence appears to offer higher sensitivity, expecially in picking up early stages of neurodegeneration.

## Cells
To visualize the morphology of cultured mouse neurons (dorsal root ganglion neurons, cortical neurons), cells were genetically modified to express a cytoplasmic fluorescent marker (EGFP, mCherry). 

## Code
The code base was written in Python (version 3.7.9) and R (version 4.4.1).





