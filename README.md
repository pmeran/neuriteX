# neuriteX
A collection of functions to quantify the integrity of neural connections (neurites) in microscopy images

## Description
For images of neurons cultured in vitro, a score is computed that reflects the integrity and health of neural connections, or neurites. The term neurites subsumes all connections emanating from neurons, i.e. axons (for outgoing or efferent signals) and dendrites (for incoming or afferent signals). The code is primarily intended for biomedical labs in academia and industry that study neurodegeneration.

## Images
For code development, 8-bit grayscale images in .png format were used, with a size of 1196 x 1196 pixels and a pixel side length corresponding to 0.69 um (mouse axons typically have a diameter of 0.6-1.1 um, but can form larger bundles). Other image formats and sizes may be used, and internal parameters (e.g. average axon thickness in pixels) can be modified.

Images were acquired for cultured mouse neurons expressing a fluorescent marker that fills out the cytoplasm and conveys the morphology of neurons and their arborizations. Based on limited tests, the provided code also produces meaningful results for (inverted) phase contrast images, which have traditionally been used in neurodegeneration research; however, fluorescence appears to offer higher sensitivity, especially in picking up early stages of neurodegeneration.

## Code
The code base was written in Python (version 3.7.9) and R (version 4.4.1).





