# neuriteX
A collection of functions to quantify the integrity of neural connections (neurites) in microscopy images

## Description
For images of neurons cultured in vitro, a score is computed that reflects the integrity and health of neural connections, or neurites. The term neurites subsumes all connections emanating from neurons, i.e. axons (for outgoing or efferent signals) and dendrites (for incoming or afferent signals). The code is primarily intended for biomedical labs in academia and industry that study neurodegeneration.

## Images
For code development, 8-bit grayscale images in .png format were used, with a size of 1196 x 1196 pixels and a pixel side length corresponding to 0.663 um (mouse axons typically have a diameter of 0.6-1.1 um, but can form larger bundles). Other image formats and sizes may be used but may require modification of internal parameters (e.g. average axon thickness in pixels). The provided code also produces meaningful results for inverted phase contrast images, which have traditionally been used in neurodegeneration research. However, fluorescence appears to offer higher sensitivity, expecially in picking up early stages of neurodegeneration.

## Cells
To visualize the morphology of cultured mouse neurons (dorsal root ganglion neurons, cortical neurons), cells were genetically modified to express a cytoplasmic fluorescent marker (EGFP, mCherry). 

## Code
The code base was written in Python (version 3.7.9) and R (version 4.4.1).





