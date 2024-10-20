# neuriteX
A collection of functions to quantify the integrity of neurons and neurites from microscopy images

## Description
The provided code takes fluorescent images of neurons cultured in vitro, and computes a score that reflects the integrity and health of neurites. The term neurites subsumes all connections emanating from neurons, i.e. both axons (carrying efferent signals) and dendrites (afferent signals).

### Images
Input images are typically grayscale images of cultured mammalian neurons (mouse, human) that express a fluorescent marker. Image depth must strictly be 8-bit grayscale (i.e. 256 gray shades for each pixel). The standard image format is .png, however other image formats may be used, or can easily be converted to .png.
The code was developed with images of size 1196 px x 1196 px, with a pixel side length corresponding to 0.69 um (mouse axons typically have a diameter of 0.6-1.1 um, but form bundles of varying size). There is no requirement for a specific image size and resolution, and parameters (e.g. average axon thickness) can be modified.

Traditionally, phase contrast images have been used to study neurodegenerative processes with cultured neurons. Limited tests suggest that the provided code also produces meaningful results with (inverted) phase contrast images, but fluorescence appears to be more sensitive in picking up early stages of neurodegeneration.






