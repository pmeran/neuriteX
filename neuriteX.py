
# © 2024 Paul Meraner
# neuriteX.py

import cv2
import numpy as np
import pandas as pd
import os, glob

from matplotlib import pyplot as plt # required for cv2

from scipy.signal import find_peaks
from scipy.signal import fftconvolve # for sgolay2d()
from scipy.ndimage import rotate # for image rotation

import random

import thinning
from tifffile import imwrite

from joblib import Parallel, delayed
import psutil  # get logical cpu count with psutil.cpu_count()



# METHODS

def p_inside_margin(sh, m, p):
    h,w = sh # image shape
    b0 = p[0] >= m
    b1 = p[0] < h-m
    b2 = p[1] >= m
    b3 = p[1] < w-m
    return b0 & b1 & b2 & b3
# end of: p_inside_margin()


def resizeImg_nearest(im, f):
    h, w = im.shape
    im_out = cv2.resize(im, (int(w * f), int(h * f)), interpolation=cv2.INTER_NEAREST)
    return im_out
# end of resizeImg_nearest ()


def resizeImg_cubic(im, f):
    h, w = im.shape
    im_out = cv2.resize(im, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)
    return im_out
# end of resizeImg_cubic ()


def resizeImg_lanczos4(im, f):
    h,w = im.shape
    im_out = cv2.resize(im, (int(w * f), int(h * f)), interpolation=cv2.INTER_LANCZOS4)
    return im_out
# end of resizeImg_lanczos4 ()


def x_generate_surface_alpha_array(im, sl):
    # 08/10/24
    # horizontal scanning (x axis)
    # generates intensity profile slope angles (in radians) for each pixel, along x axis
    # im = image as numpy array
    # sl = slope correction factor
    dif = np.diff(im.astype(int), n=1, axis=1) # width shortened by 1 px
    dif = dif * sl
    alf = np.arctan2(dif, 1)

    aPd = np.zeros_like(im).astype(float)
    D = np.dstack([alf[:,0:-1], alf[:,1:]])
    Dav = np.average(D, axis=2)
    aPd[:,1:-1] = Dav

    return aPd
# end of: x_generate_surface_alpha_array()


def y_generate_surface_alpha_array(im, sl):
    # 08/10/24
    # vertical scanning (y axis)
    # generates intensity profile slope angles (in radians) for each pixel, along y axis
    # im = image as numpy array
    # sl = slope correction factor
    dif = np.diff(im.astype(int), n=1, axis=0) # height shortened by 1 px
    dif = dif * sl
    alf = np.arctan2(dif, 1)

    aPd = np.zeros_like(im).astype(float)
    D = np.dstack([alf[0:-1,:], alf[1:, :]])
    Dav = np.average(D, axis=2)
    aPd[1:-1, :] = Dav

    return aPd
# end of: y_generate_surface_alpha_array()


def generate_imgGradients (im, **kwargs):
    # 08/13/24 im = small image, resized by factor of 3 to 5 (before being passed to this function)
    # im = image
    # slF = slope correction factor

    # generate arrays of surface slope angles (in radians)
    slF = kwargs.get('slopeF', 0.2)  #

    xaPd = x_generate_surface_alpha_array(im, slF)
    yaPd = y_generate_surface_alpha_array(im, slF)

    # generate lenPd (xy projection of unitary perpendicular (!) surface vector; lenPd.min() = 0, lenPd.max() < 1)
    # generate dirPd (direction of xy projection of unitary perpendicular surface vector; range [0, 2*pi)
    xlenPd = -np.sin(xaPd)
    ylenPd = -np.sin(yaPd)
    lenPd = np.sqrt((xlenPd ** 2 + ylenPd ** 2) / 2)

    dirPd = np.arctan2(-ylenPd, xlenPd) % (2 * np.pi)  # tricky - sign, np.arctan2 format (y,x) !!
    dirPd[lenPd == 0] = -1

    return lenPd, dirPd
# end of: generate_imgGradients()


def generate_x_peakArray_5 (im, **kwargs):
    # 08/17/24 scans cross-sectional profiles (along x axis) for peaks
    # output: 2-tuple with y (rows) and x (columns) coords (format like np.where)

    # 08/26/24 can handle all sorts of im inputs, including:
    #     - "im" is large vector length projection pad (named "lenPad" or similar) from large, 3 or 5 times expanded image
    #     - "im" values are in range [0, 1) i.e. can be 0, never reach 1
    #     - function identifies projection vector length peaks
    # Typical argument lineup:
    # generate_x_peakArray_5(lenPadS, heightMin = 0, promMin=0.1, promMax=1, widthMin=1, widthMax = 10, distMin=1, heightRel = 0.2)
    # output: peak positions, given in two formats: as array "pkAr" or binary image "pkPd" (numpy array np.uint8)

    # im is image in numpy array np.uint8 format
    # im is assumed to be expanded 3 or 5 times, and parameter defaults are chosen
    #       accordingly (though code should work fine with different scaling factors)

    # flatten list of lists
    def flatten(xss):
        return [x for xs in xss for x in xs]

    h_im, _ = im.shape

    heightMin = kwargs.get('heightMin', 0)  # heightMin was 5
    promMin = kwargs.get('promMin', 5)  # the smaller the more sensitive the noisier !
    promMax = kwargs.get('promMax', 255)
    widthMin = kwargs.get('widthMin', 3)  # consider possible scaling by factor 3 or 5
    widthMax = kwargs.get('widthMax', 25)  # consider possible scaling by factor 3 or 5
    distMin = kwargs.get('distMin', 3)
    heightRel = kwargs.get('heightRel', 0.5)

    x_list = []
    y_list = []

    for ir in range(1, h_im - 1):
        # maxima
        peaks, proms = find_peaks(im[ir, :], height=heightMin, prominence=[promMin, promMax],
                                  width=[widthMin, widthMax], distance=distMin, rel_height=heightRel)
        len_peaks = len(peaks)
        if len_peaks > 0:
            ips_peaks = np.average((proms['left_ips'], proms['right_ips']), axis=0).astype(int)
            x_list.append(list(ips_peaks))
            y_list.append([ir] * len_peaks)

    y_list = flatten(y_list)
    x_list = flatten(x_list)

    pkAr = (np.array(y_list, dtype=np.int64), np.array(x_list, dtype=np.int64))

    pkPd = np.zeros_like(im, dtype=np.uint8)
    pkPd[pkAr] = 255

    return pkAr, pkPd # pkAr = peaks in arrays as 2-tuple, pkPd = peaks in image
# end of: generate_x_peakArray_5 (im, **kwargs)


def neuriteSegmentation_X_4 (imE, **kwargs):

    # 09/07/24 filter adjustment, now using pixel_dirArr_2(), isNeurite_4()
    # 08/30/24 06:45 pm new version 3
    #     main modification: dirAr vertically extended to rows p[0] -6 to +6 (i.e. 13 rows total)
    #     (in previous versions: rows p[0] -3 to +3 i.e. 7 rows total
    #
    # input: image 'imE' np.uint8, assumed to be scaled by factor 3 or 5 (not scaled or rotated inside this function)
    # Function scans for peaks along horizontal (x-axis) profile lines, segments peaks as neurites or non-neurites
    # returns: [count, count_px, count_neur], neurAr
    #     count (number of all peaks)
    #     count_px (number of all peaks, excluding those close to image border),
    #     count_neur (number of peaks classified as neurites),
    #     neurAr (array with N_neur rows, 2 columns, representing positions of neurite pixels)

    # **kwargs
    # t = kwargs.get('t', min(len(T), 5000)) # see further below

    def pixel_dirArr_2(p, **kwargs):
        """
        Notes 10/19/24
        returns dirAr (shape (7,2)) with vector projection directions on left and right hand side of pixel p
        dirAr may also contain values of -1 (maximum number of allowed -1 values is determined by variable 'missing')
        """
        # NOTE 09/05/24 THIS VERSION _2 looks at 7 x 21 subimage (vs 13 x 21 for later versions)
        #       mask less stringent
        # created 08/30/24
        # NOTE 08/30/24 this version _2 relaxes conditions to increase sensitivity
        # Relies on finding variables 'lenPeakPadE' and 'dirPadE' in superordered scope ! (previous names x_lenPeakPadS, dirPadS)
        # By placing 'pixel_dirArr' into 'neuriteSegmentation_X', lenPeakPadE and dirPadE are directly accessible
        # ONLY use the 'global' keyword if lenPeakPadE and dirPadE are defined in the global scope

        # input p = coordinates (as 2-tuple) for pixel of interest

        # output 'dirAr' is array with dirAr.shape = (7,2), contains vector projection directions left and right of pixel p
        # arbitrary size of small observation window is (h,w) = (2*hH + 1, 2*wH + 1)

        missing = kwargs.get('missing', 2) # maximum number of missing values in xAr below

        hH = 3  # h for height, H for half edge
        wH = 10

        hF = (2 * hH) + 1  # h for height, F for full edge
        wF = (2 * wH) + 1  # w for width, F for full edge

        scT = lenPeakPadE[p[0] - hH:p[0] + hH + 1, p[1] - wH:p[1] + wH + 1]

        mask = np.ones((hF, wF), dtype=np.uint8) * 255
        mask[:,[10]] = 0 # new 09/06/24, relaxes filter considerably (ca. 60% of peaks pass before isNeurite)

        scT_filt = np.bitwise_and(scT, mask)

        xAr = np.array([[-1, -1]] * hF, dtype=int)
        dirAr = np.array([], dtype=float)

        filled_L = [] # previously countL
        filled_R = [] # previously countR

        for i in range(hF):
            wL = np.where(scT_filt[i, 0:wH] == 255)[0]
            wR = np.where(scT_filt[i, wH:] == 255)[0]
            if (len(wL) > 0):
                xAr[i, 0] = wL[-1]
                filled_L.append(i)
            if (len(wR) > 0):
                xAr[i, 1] = wR[0] + wH
                filled_R.append(i)

        filled_LR = np.intersect1d(filled_L, filled_R) # filled_LR is numpy array

        if len(filled_LR) >= hF - missing:
            ULp = (p[0] - hH, p[1] - wH)

            w = np.where(xAr > -1)
            xpos = xAr[w] + ULp[1]  # xAr[w] is np.array of dtype int32
            ypos = w[0] + ULp[0]

            dirVals = dirPadE[(ypos, xpos)]

            dirAr = np.array([[-1, -1]] * hF, dtype=float)
            dirAr[w] = dirVals

        return dirAr
        # NOTE 08/30/24 dirAr may contain -1
        # dirAr is array with dirAr.shape = (7,2), contains vector projection directions
        #     on left and right hand side of pixel p
    # end of: pixel_dirArr_2()


    def isNeurite_4(dirAr):

        """
        Notes 10/19/24
        isNeurite_4 functions as a filter to determine if dirAr is compatible with neurite

        Crit_10 (boolean)
        determines if left vectors compared to each other point in similar direction, and if
            right vectors compared to each other point in similar direction
        Cin_5 (boolean)
        inclusion criterion (tests how parallel vectors are in same row i.e. how parallel opposite vectors are)

        if Crit_10 and Cin_5:
            out = True
        """
        # 08/30/24 new version - accepts dirAr containing some -1 values (i.e. missing dir values)

        # Cin = inclusion criterion (must be True)
        # Cex = exclusion criterion (must be False)

        out = False

        if len(dirAr) > 0:

            sL = np.where(dirAr[:,0] != -1)[0]
            sR = np.where(dirAr[:,1] != -1)[0]

            arL = dirAr[sL, 0]
            arR = dirAr[sR, 1]
            arRr = (arR + np.pi) % (2 * np.pi) # rotating right column of directions by pi, or 180 degrees

            # Crit_10 is True if proper left vectors are very close, and proper right vectors are very close
            Crit_10 = np.std(arL) < 0.4 and np.std(arRr) < 0.4
            # works kind of ok, should probably be included
            # examples (comparing largely healthy / mostly degenerated neurites)
            # < 0.4: 37.49 / 12.81
            # < 0.3: 28.44 /  6.52
            # examples Crit_10 0.2 -> 16.06/3.81, Crit_10 0.3 -> 28.53/9.84

            dif_arL = np.diff(arL)
            dif_arRr = np.diff(arRr)

            sLR_ = np.intersect1d(sL, sR)

            arL_ = dirAr[sLR_, 0]
            arR_ = dirAr[sLR_, 1]
            arRr_ = (arR_ + np.pi) % (2 * np.pi)

            # 09/06/24 Cin_5 - best criterion !
            # Cin_5 tests parallelism of vectors in same row in arL_ and arRr_
            Cin_5 = np.percentile(abs(np.subtract(arL_, arRr_)), 70) < 0.5
            # comparing neurites halthy / degenerated
            # perc 80, < 0.4 -> 20.95 / 3.76
            # perc 80, < 0.5 -> 27.97 / 6.04
            # perc 90, < 0.5 -> 21.7  / 3.83 , 22.41 / 3.25 with another pair of pictures
            # perc 70, < 0.5 -> 34.84 / 8.07
            # perc 95, < 0.5 -> 20.39 / 2.82

            # combo Crit_10 (< 0.4) and Cin_5 (perc 80, < 0.5) -> 24.27 / 3.04
            # 09/07/24 What filters to use ?
            # Cin_5 is best criterion (left/right parallelism), can possibly be usd alone
            # Crit_10 (uniformity left, uniformity right) makes sense, combine with Cin_5
            # Cex_5 (centrifugality) was removed, somehow redundant with Cin_5

            if Crit_10 and Cin_5:
                out = True

        return out
    # end of: isNeurite_4()

    count = 0
    neurAr = np.empty((0,2), dtype=np.uint16)

    # in image 'imE', find peaks corresponding to cross sections of neurites or blebs/blobs
    peakArE, peakPadE = generate_x_peakArray_5(imE, promMin=16, promMax=255, widthMin=1, widthMax = 30, distMin = 1, heightRel = 0.1)
        # NOTE peaks result from horizontal (x-axis) profile scanning
    lenPadE, dirPadE = generate_imgGradients(imE, slopeF = 0.2)  # was previously: lenPadS, dirPadS = generate_gradients_S(imgE)
    # find peaks in orthogonal vector projection length pad 'lenPadE'
    _, lenPeakPadE = generate_x_peakArray_5(lenPadE, heightMin=0, promMin=0.1, promMax=1, widthMin=1, widthMax=15, distMin=1, heightRel=0.1)
    # NOTE 09/06/24 promMin was 0.05 -> now 0.1
    T = list(tuple(zip(peakArE[0], peakArE[1])))

    if len(T) > 0:
        random.shuffle(T)  # modifies T by reference
        t = kwargs.get('t', min(len(T), 5000))
        peakList = []
        neurList = []
        for px in T[0:t]:
            count += 1
            if p_inside_margin(peakPadE.shape, 12, px):
                peakList.append(px)
                dirArr = pixel_dirArr_2(px, missing=3)  # uses lenPeakPadE, dirPadE as variables
                N = isNeurite_4(dirArr)
                if N:
                    neurList.append(px)
        peakAr = np.array(peakList, dtype=np.uint16)
        neurAr = np.array(neurList, dtype=np.uint16)
    else:
        peakAr = np.empty((0,2), dtype=np.uint16)
        neurAr = np.empty((0,2), dtype=np.uint16)

    return [len(T), count, len(peakAr), len(neurAr)], peakAr, neurAr, lenPadE, dirPadE, lenPeakPadE
    # len(T) is the number of all detected cross-sectional peaks
    # count is the number of cross-sectional peaks used for neurite segmentation
    #   if t is specified under **kwargs: count = min(len(T), t)
    #   otherwise: count = min(len(T), 5000)
    # len(peakAr) is number of peaks used for neurite segmentation, minus peaks close to edges
    # len(neurAr) is number of peaks classified as neurites
# end of: neuriteSegmentation_X_4()

# img0_sg = sgolay2d(img0, 5, 3)
def sgolay2d(z, window_size, order, derivative=None):
    """
    added 09/07/24
    great performance !
    https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    """
    # 09/07/24
    # from scipy.signal import fftconvolve

    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size ** 2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size ** 2, )

    # build matrix of system of equation
    A = np.empty((window_size ** 2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs(np.flipud(z[1:half_size + 1, :]) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(z[-half_size - 1:-1, :]) - band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(z[:, 1:half_size + 1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(z[:, -half_size - 1:-1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(z[1:half_size + 1, 1:half_size + 1])) - band)
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(np.flipud(np.fliplr(z[-half_size - 1:-1, -half_size - 1:-1])) - band)

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(np.flipud(Z[half_size + 1:2 * half_size + 1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(np.fliplr(Z[-half_size:, half_size + 1:2 * half_size + 1]) - band)

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode='valid'), fftconvolve(Z, -c, mode='valid')
# end of: sgolay2d()


# nX_correction (im, promMin = 5, perc_hi = 90, lim_I_new = 180, gamma = 0.7, win = 3, ord = 2)
def nX_correction (im, **kwargs):

    # 12/02/24 modification of imageCorrection_new_3
    # 09/14/24 modification - includes sgolay2d noise removal step
    # 09/04/24
    # NOTE 09/05/24 consider:
    # Amplification of signal intnsity seems to rather decrease N_perc, maybe lim_I_new should be 150 to 200
    # background correction removed without detrimental effect (actually close to no effect at all)
    # NOTE to further enhance resulting image intensity, DECREASE perc_hi
    # NOTE 09/06/24 no peaks may be detected in very low intensity images -> adjust promMin
    # lim_I_new = 180; perc_hi = 90; gamma = 0.7

    # promMin = 5; perc_hi = 90; lim_I_new = 180; gamma = 0.7; win = 5; ord = 2

    promMin = kwargs.get('promMin', 5) # minimum peak prominence in original image
    perc_hi = kwargs.get('perc_hi', 90) # percentile of peak intensities chosen as pivot point
    lim_I_new = kwargs.get('lim_I_new', 180) # output intensity pivot
    gamma = kwargs.get('gamma', 0.7) # gamma correction parameter
    win = kwargs.get('win', 3) # Savitzky-Golay window size
    ord = kwargs.get('ord', 2) # Savitzky-Golay order

    # find peaks along horizontal (x) scan lines
    pkTu_x, _ = generate_x_peakArray_5(im, promMin=promMin, promMax=255, widthMin=1, widthMax = 6, distMin = 1, heightRel = 0.1)
    # find peaks along vertical (y) scan lines
    im_tr = np.transpose(im)
    pkTu_tr, _ = generate_x_peakArray_5(im_tr, promMin=promMin, promMax=255, widthMin=1, widthMax = 6, distMin = 1, heightRel = 0.1)
    pkTu_y = (pkTu_tr[1], pkTu_tr[0])

    pkL = list(zip(np.hstack((pkTu_x[0], pkTu_y[0])), np.hstack((pkTu_x[1], pkTu_y[1]))))
    ex_sta = 1

    if len(pkL) == 0: # if no peaks found, output_image = input_image, exit_status = -1
        im_out = im
        ex_sta = -1
    else:
        pkL = list(set(pkL)) # remove duplicate pixels
        pkAr = np.array(pkL)
        pkTu = (pkAr[:,0], pkAr[:,1])

        bwPd = np.zeros_like(im)
        bwPd[pkTu] = 255
        # cv2.imshow('bwPd', bwPd)
        bwPd_thin = thinning.guo_hall_thinning(bwPd)
        # cv2.imshow('bwPd_thin', bwPd_thin)

        pkTu2 = np.where(bwPd_thin==255)
        pkI = im[pkTu2] # intensity values for peak pixels
        pkI_perc_hi = np.percentile(pkI, perc_hi)

        im_out = im.astype(np.float64) # conversion np.uint8 to np.float64
        im_out = im_out * lim_I_new/pkI_perc_hi # normalize with lim_I_new as pivot

        im_out = 255 * np.power((im_out / 255), gamma)  # gamma correction

        # 2D Savitzky-Golay noise reduction filter | sgolay2d(z, window_size, order, derivative=None):
        im_out = sgolay2d(im_out, win, ord, None)

        # clip, round, convert to np.uint8
        im_out = np.rint(np.clip(im_out, 0, 255)).astype(np.uint8)

    return im_out, ex_sta
# end of: nX_correction (im, **kwargs)


# D = nX_segmentation (im, ptUL=(10,10), eH=300, extF=3, win=3, ord=2, t=100000)
def nX_segmentation (im, im_name, **kwargs):

    # 12/02/24 modification of neuriteSegmentation_v5()
    # 10/19/24 new version - with mx, ord (arguments for sgolay2)
    # based on neuriteSegmentation_Full_4_Demo_server()
    # input image is corrected (with imageCorrection_new_3)

    ptUL = kwargs.get('ptUL', (0,0)) # upper left anchor point
    eH = kwargs.get('eH', 100) # half edge of image subsquare to analyze
    extF = kwargs.get('extF', 3) # image scaling factor
    win = kwargs.get('mx', 3) # Savitzky-Golay window size
    ord = kwargs.get('ord', 2)# Savitzky-Golay order
    t = kwargs.get('t', 5000) # upper limit of random peaks to analyze

    eF = (2*eH)+1
    # reduce eF if too large:
    h,w = im.shape
    eF = min(min(h - ptUL[0], eF), min(w - ptUL[1], eF))
    if eF % 2 == 0:
        eF = eF - 1

    im_s = im[ptUL[0]:ptUL[0] + eF, ptUL[1]:ptUL[1] + eF]

    imE = cv2.resize(im_s, (eF * extF, eF * extF), interpolation=cv2.INTER_LANCZOS4)

    # tested - denoising with Savitzky-Golay works better than applying Gaussian blur
    imE0 = np.rint(np.clip(sgolay2d(imE, win, ord, None), 0, 255)).astype(np.uint8)

    imE30 = rotate(imE, angle=30, reshape=False)
    imE30 = np.rint(np.clip(sgolay2d(imE30, win, ord, None), 0, 255)).astype(np.uint8)

    imE60 = rotate(imE, angle=60, reshape=False)
    imE60 = np.rint(np.clip(sgolay2d(imE60, win, ord, None), 0, 255)).astype(np.uint8)

    imE90  = np.flipud(np.transpose(imE0))  # rotate 90 degrees counterclockwise
    imE120 = np.flipud(np.transpose(imE30)) # rotate 90 degrees counterclockwise
    imE150 = np.flipud(np.transpose(imE60)) # rotate 90 degrees counterclockwise

    # print('Starting neuriteSegmentation_X_4...')
    C0, _, _, _, _, _ = neuriteSegmentation_X_4(imE0, t=t)
    # print('C0 done...')
    C30, _, _, _, _, _ = neuriteSegmentation_X_4(imE30, t=t)
    # print('C30 done...')
    C60, _, _, _, _, _ = neuriteSegmentation_X_4(imE60, t=t)
    # print('C60 done...')
    C90, _, _, _, _, _ = neuriteSegmentation_X_4(imE90, t=t)
    # print('C90 done...')
    C120, _, _, _, _, _ = neuriteSegmentation_X_4(imE120, t=t)
    # print('C120 done...')
    C150, _, _, _, _, _ = neuriteSegmentation_X_4(imE150, t=t)
    # print('C150 done...')

    N_perc_list = []
    for C in [C0, C30, C60, C90, C120, C150]:
        if C[2] > 0:
            N_perc_list.append(100 * C[3]/C[2])
        else:
            N_perc_list.append(-1)

    N_perc_list_nonzero = [p for p in N_perc_list if p > -1]
    if len(N_perc_list_nonzero) > 0:
        N_perc = np.average(np.array(N_perc_list_nonzero))
    else:
        N_perc = -1

    D = {'img_file': im_name, 'C0': C0, 'C30': C30, 'C60': C60, 'C90': C90, 'C120': C120, 'C150': C150, 'N_perc_list': N_perc_list, 'N_perc': N_perc}

    # What the four numbers in C0 mean (same applies to C30, C60, etc.):
    #   C0[0] = len(T) is the number of all detected cross-sectional peaks
    #   C0[1] = count is the number of cross-sectional peaks used for neurite segmentation
    #       if t is specified under **kwargs: count = min(len(T), t)
    #       otherwise: count = min(len(T), 5000)
    #   C0[2] = len(peakAr) is number of peaks used for neurite segmentation, minus peaks close to edges
    #   C0[3] = len(neurAr) is number of peaks classified as neurites

    # N_perc_list with six numbers lists Cx[3]/Cx[2] for C0, C30, C60, etc.
    # N_perc is average of N_perc_list

    return D #, imE0, imBW_0, imBW_30, imBW_60
# end of: nX_segmentation()


# D, stack = nX_segmentation_test (imgX, '012a.png', ptUL=(0,0), eH=100, extF = 3, win = 3, ord=2, t = 100000)
def nX_segmentation_test (im, im_name, **kwargs):

    # 12/02/24 modification of neuriteSegmentationMonitorFull
    # 10/19/24 modified to scan directions 0, 30, 60 degrees; does not have argument alf for free rotation

    # 09/06/24
    # segmentation by scanning only in two directions: alf, and alf + np.pi/2
    # saves stack of images in .tif format (can be viewed as stack with imagej/fiji)
    # based on neuriteSegmentation_Full_4_Demo
    # 09/02/24
    # 08/30/24 09:00 pm
    # 08/31/24 NOTE works kind of ok, settings can be tweaked easily
    # im is original image without corrections
    # neuriteSegmentation_Full(im, ptUL=(600,600), eH = 100, extF = 5, t = 5000)
    # ptUL = (0,0); eH = 300; eF = (2*eH)+1; extF = 5; t = 500000
    # ptUL = (500,600); eH = 150; eF = (2*eH)+1; extF = 5; t = 500000
    # alf = 10; ptUL = (500,50); eH = 150; eF = (2*eH)+1; extF = 3; t = 100000

    # alf = kwargs.get('alf', 0)
    ptUL = kwargs.get('ptUL', (0,0)) # upper left anchor point
    eH = kwargs.get('eH', 100) # half edge of image subsquare to analyze
    extF = kwargs.get('extF', 3) # scaling factor
    win = kwargs.get('win', 3) # Savitzky-Golay window size
    ord = kwargs.get('ord', 2) # Savitzky-Golay order
    t = kwargs.get('t', 5000) # upper limit of random peaks to analyze

    eF = (2*eH)+1
    # reduce eF if too large:
    h,w = im.shape
    eF = min(min(h - ptUL[0], eF), min(w - ptUL[1], eF))
    if eF % 2 == 0:
        eF = eF - 1

    # im = cv2.imread(im_path + im_fname, -1) # source image should be original, or corrected original (with same siz as original)
    im_s = im[ptUL[0]:ptUL[0] + eF, ptUL[1]:ptUL[1] + eF]

    imE = cv2.resize(im_s, (eF * extF, eF * extF), interpolation=cv2.INTER_LANCZOS4)

    ####
    imE0 = np.rint(np.clip(sgolay2d(imE, win, ord, None), 0, 255)).astype(np.uint8)  # tested - better denoise than blur !

    imE30 = rotate(imE, angle=30, reshape=False)
    imE30 = np.rint(np.clip(sgolay2d(imE30, win, ord, None), 0, 255)).astype(np.uint8)

    imE60 = rotate(imE, angle=60, reshape=False)
    imE60 = np.rint(np.clip(sgolay2d(imE60, win, ord, None), 0, 255)).astype(np.uint8)

    imE90  = np.flipud(np.transpose(imE0))  # rotate 90 degrees counterclockwise
    imE120 = np.flipud(np.transpose(imE30)) # rotate 90 degrees counterclockwise
    imE150 = np.flipud(np.transpose(imE60)) # rotate 90 degrees counterclockwise

    print('Starting segmentation...')
    print('imE0')
    C0, peakAr0, neurAr0, lenPdE0, dirPdE0, lenPeakPdE0 = neuriteSegmentation_X_4(imE0, t=t)
    print('imE30')
    C30, peakAr30, neurAr30, lenPdE30, dirPdE30, lenPeakPdE30 = neuriteSegmentation_X_4(imE30, t=t)
    print('imE60')
    C60, peakAr60, neurAr60, lenPdE60, dirPdE60, lenPeakPdE60 = neuriteSegmentation_X_4(imE60, t=t)
    print('imE90')
    C90, peakAr90, neurAr90, lenPdE90, dirPdE90, lenPeakPdE90 = neuriteSegmentation_X_4(imE90, t=t)
    print('imE120')
    C120, peakAr120, neurAr120, lenPdE120, dirPdE120, lenPeakPdE120 = neuriteSegmentation_X_4(imE120, t=t)
    print('imE150')
    C150, peakAr150, neurAr150, lenPdE150, dirPdE150, lenPeakPdE150 = neuriteSegmentation_X_4(imE150, t=t)

    N_perc_list = []
    for C in [C0, C30, C60, C90, C120, C150]:
        if C[2] > 0:
            N_perc_list.append(100 * C[3]/C[2])
        else:
            N_perc_list.append(-1)

    N_perc_list_nonzero = [p for p in N_perc_list if p > -1]
    if len(N_perc_list_nonzero) > 0:
        N_perc = np.average(np.array(N_perc_list_nonzero))
    else:
        N_perc = -1

    D = {'img_file': im_name, 'C0': C0, 'C30': C30, 'C60': C60, 'C90': C90, 'C120': C120, 'C150': C150, 'N_perc_list': N_perc_list, 'N_perc': N_perc}

    print('check_0')
    peakPd0 = np.zeros_like(imE0)
    peakPd0[(peakAr0[:,0], peakAr0[:,1])] = 255
    peakPd90 = np.zeros_like(imE90)
    peakPd90[(peakAr90[:,0], peakAr90[:,1])] = 255
    peakPd90 = np.fliplr(np.transpose(peakPd90))
    peakPd0 = cv2.bitwise_or(peakPd0, peakPd90)
    del peakPd90

    print('check_1')
    peakPd30 = np.zeros_like(imE30)
    peakPd30[(peakAr30[:,0], peakAr30[:,1])] = 255
    peakPd120 = np.zeros_like(imE120)
    peakPd120[(peakAr120[:,0], peakAr120[:,1])] = 255
    peakPd120 = np.fliplr(np.transpose(peakPd120))
    peakPd30 = cv2.bitwise_or(peakPd30, peakPd120)
    del peakPd120

    print('check_2')
    peakPd60 = np.zeros_like(imE60)
    peakPd60[(peakAr60[:, 0], peakAr60[:, 1])] = 255
    peakPd150 = np.zeros_like(imE150)
    peakPd150[(peakAr150[:, 0], peakAr150[:, 1])] = 255
    peakPd150 = np.fliplr(np.transpose(peakPd150))
    peakPd60 = cv2.bitwise_or(peakPd60, peakPd150)
    del peakPd150

    print('check_3')
    neurPd0 = np.zeros_like(imE0)
    neurPd0[(neurAr0[:, 0], neurAr0[:, 1])] = 255
    neurPd90 = np.zeros_like(imE90)
    neurPd90[(neurAr90[:, 0], neurAr90[:, 1])] = 255
    neurPd90 = np.fliplr(np.transpose(neurPd90))
    neurPd0 = cv2.bitwise_or(neurPd0, neurPd90)
    del neurPd90

    print('check_4')
    neurPd30 = np.zeros_like(imE30)
    neurPd30[(neurAr30[:, 0], neurAr30[:, 1])] = 255
    neurPd120 = np.zeros_like(imE120)
    neurPd120[(neurAr120[:, 0], neurAr120[:, 1])] = 255
    neurPd120 = np.fliplr(np.transpose(neurPd120))
    neurPd30 = cv2.bitwise_or(neurPd30, neurPd120)
    del neurPd120

    print('check_5')
    neurPd60 = np.zeros_like(imE60)
    neurPd60[(neurAr60[:, 0], neurAr60[:, 1])] = 255
    neurPd150 = np.zeros_like(imE150)
    neurPd150[(neurAr150[:, 0], neurAr150[:, 1])] = 255
    neurPd150 = np.fliplr(np.transpose(neurPd150))
    neurPd60 = cv2.bitwise_or(neurPd60, neurPd150)
    del neurPd150

    print('check_6')
    lenPeakPd0 = np.bitwise_or(lenPeakPdE0, np.fliplr(np.transpose(lenPeakPdE90)))
    lenPeakPd30 = np.bitwise_or(lenPeakPdE30, np.fliplr(np.transpose(lenPeakPdE120)))
    lenPeakPd60 = np.bitwise_or(lenPeakPdE60, np.fliplr(np.transpose(lenPeakPdE150)))

    stack_ = np.stack((imE0, peakPd0, neurPd0, lenPeakPd0, imE30, peakPd30, neurPd30, lenPeakPd30,
                       imE60, peakPd60, neurPd60, lenPeakPd60), axis=0)
    return D, stack_
    # imwrite('./stacks/' + 'blebsSimul_01_stack.png', stack, photometric='minisblack') # imwrite from tifffile
# end of: nX_segmentation_test()

# END OF METHODS


#########################################
#########################################
#########################################
#########################################
#########################################
#########################################


###################################
# 1. Single image analysis

path_main = '<path_main>/neuriteX_root_folder/'
os.chdir(path_main)

if not os.path.exists('./img_corr/'):
    os.makedirs('./img_corr/')

if not os.path.exists('./img_stack'):
    os.makedirs('./img_stack/')

path_ori = './img_ori/'
path_corr = './img_corr/'
path_stack = './img_stack/'

img_files = sorted([os.path.basename(x) for x in glob.glob('{}*.png'.format(path_ori))])
img_file = img_files[0]

# 1.1 Image display
imgX = cv2.imread(path_ori + img_file, -1)
cv2.imshow('imgX', imgX)


# 1.2 Image correction
# Run time 5-10 sec
imgC, status = nX_correction(imgX, promMin = 5, perc_hi = 90, lim_I_new = 180, gamma = 0.9, win = 3, ord = 2)
cv2.imshow('imgC', imgC)


# 1.3 Image segmentation and generation of test images
# NOTE slow process
# ca. 15 sec for eH = 100 (image area to analyze 201 px x 201 px)
# ca. 6 min for eH = 580 (image area to analyze 1161 px x 1161 px)
D, stack = nX_segmentation_test (imgC, img_file, ptUL=(10,10), eH=100, extF = 3, win = 3, ord=2, t = 200000)
imwrite(path_stack + 'stack_' + img_file, stack, photometric='minisblack')  # imwrite from tifffile


# 1.4 Image segmentation
D = nX_segmentation (imgC, img_file, ptUL=(10,10), eH=100, extF=3, win=3, ord=2, t=100000)

# end of: 1. Single image analysis
##########################################


##########################################
# 2. Batch analysis of multiple images

# 2.1 Batch image correction

# path_main = '<path_main>/neuriteX_root_folder/'
os.chdir(path_main)

n_j = 8 # process number, must be identical to cpu count in 'SBATCH -c '

path_ori   = './img_ori/'
path_corr  = './img_corr/'

img_list = [os.path.basename(x) for x in glob.glob('{}*.png'.format(path_ori))]
img_list = sorted(img_list)

def corr_parallel(im_fname):
    print(im_fname)
    im = cv2.imread(path_ori + im_fname, -1)
    im_corr, exit_status = nX_correction(im, promMin = 5, perc_hi = 95, lim_I_new = 180, gamma = 0.8, win = 3, ord = 2)
    im_fname_new = im_fname.replace('.png', '_corr.png')
    cv2.imwrite(path_corr + im_fname_new, im_corr)
    return [im_fname_new, exit_status]

results_corr = Parallel(n_jobs = n_j)(delayed(corr_parallel)(img_fname) for img_fname in img_list)

df_corr = pd.DataFrame(results_corr)
df_corr.to_pickle('./df_corr.pkl')

# end of: 2.1 Batch analysis - image correction
###############################################


###############################################
# 2.2 Batch analysis - image segmentation
# processing time for all images (42) in img_ori, eH = 200 ->  ~25 min

# path_main = '<path_main>/neuriteX_root_folder/'
os.chdir(path_main)

n_j = 8 # process number, must be identical to cpu count in 'SBATCH -c '

path_corr = './img_corr/'

img_list = [os.path.basename(x) for x in glob.glob('{}*.png'.format(path_corr))]
img_list = sorted(img_list)

def seg_parallel(im_fname):
    print(im_fname)
    im = cv2.imread(path_corr + im_fname, -1)
    D = nX_segmentation(im, im_fname.replace('_corr.png', '.png'), ptUL=(100,100), eH=100, extF=3, win=3, ord=2, t=100000)
    return D

results_seg = Parallel(n_jobs = n_j)(delayed(seg_parallel)(img_fname) for img_fname in img_list)

df_seg = pd.DataFrame(results_seg)
df_seg.to_pickle('./df_seg.pkl')

# end of: 2.2 Batch analysis - image segmentation
#################################################

#################################################
# Create excel spreadsheet

import cv2
import numpy as np
import pandas as pd
import os, sys, glob

# path_main = '<path_main>/neuriteX_root_folder/'
os.chdir(path_main)

# path_ori   = './img_ori/'
# path_corr  = './img_corr/'

#   3 scenes uncut - 3 scenes cut
#   6 scenes/well, 1 well -> 6 scenes total
#   6 scenes x 7 time points -> 42 images

condition_list = ['no-insert'] * 42
img_list = sorted([os.path.basename(x) for x in glob.glob('{}*.png'.format('./img_ori/'))])
well_list = ['B1'] * 42

state_list = ((['uncut'] * 7 * 3) + (['cut'] * 7 * 3))
time_s_list = ['4 h','7 h','10 h','13 h','16 h','24 h','28 h'] * 6
time_n_list = [4,7,10,13,16,24,28] * 6

df = pd.DataFrame({'idx_zero': list(range(0, len(img_list))), 'idx_one': list(range(1, len(img_list)+1)), 'well': well_list,
    'file': img_list, 'condition': condition_list, 'state': state_list, 'time_s': time_s_list, 'time_n': time_n_list})
df.to_excel('./df_excel.xlsx', sheet_name='drg', index=False)

# end of: Create excel spreadsheet
################################


################################
# 3. Merging neurite integrity scores with experimental metadata

# 3.1 Generate df_R1.csv with 'N_perc'

import cv2
import numpy as np
import pandas as pd
import os, sys, glob

# path_main = '<path_main>/neuriteX_root_folder/'
os.chdir(path_main)

df_seg = pd.read_pickle('./df_seg.pkl')
df_excel = pd.read_excel('./df_excel.xlsx')

df_R1 = pd.concat([df_seg, df_excel], axis=1)
df_R1 = df_R1.loc[:, ['file','img_file', 'well', 'time_n', 'state', 'N_perc', 'condition']]

df_R1.to_csv('./df_R1.csv')
# check NP_perc

# 3.2 Generate df_R2.csv with 'N_perc_corr'
# Data correction:
# 1. For each scene, calculate sum of NP_perc, then average of sums for 'scenes uncut' and 'scenes cut'
# 2. Correct data by multiplication (addition has been tried but doesn not work as well)

df_R2 = df_R1.copy()

# create column 'scene'
df_R2['scene'] = [f[0:3] for f in df_R2.file]
# group by well, state, scene - then calculate sum of N_perc over all time points, and store in new variable 'sum_ref'

df_R2['sum_ref'] = df_R2.groupby(['well', 'state', 'scene'])['N_perc'].transform('sum')

# calculate mean of 'sum_ref' for each well and state
df_R2['sum_ref_mean'] = df_R2.groupby(['well', 'state'])['sum_ref'].transform('mean')
# correct N_perc values
df_R2['N_perc_corr'] = df_R2.N_perc * df_R2.sum_ref_mean/df_R2.sum_ref

df_R2.to_csv('df_R2.csv')


#############################################################
# 3.2 Generate df_R2.csv with 'NII_well'
df_R3 = df_R2.copy()

# create df_R3['NII_well'] - normalization for groups ['well', 'state']
# Normalize based on first time point (4 h) for each state (separate normalization for 'uncut' and 'cut')

df_R3['NII_well'] = 0
for well in df_R3.well:
    b0 = df_R3.well == well
    b1 = df_R3.time_n==4
    b_cut = df_R3.state=='cut'
    b_uncut = df_R3.state=='uncut'

    w_ref_cut = np.average(df_R3.loc[b0 & b1 & b_cut, 'N_perc_corr'])
    w_ref_uncut = np.average(df_R3.loc[b0 & b1 & b_uncut, 'N_perc_corr'])

    df_R3.loc[b0 & b_cut, 'NII_well'] = df_R3.loc[b0 & b_cut, 'N_perc_corr']/w_ref_cut
    df_R3.loc[b0 & b_uncut, 'NII_well'] = df_R3.loc[b0 & b_uncut, 'N_perc_corr']/w_ref_uncut

df_R3.to_csv('df_R3.csv')

# end of: create .csv file
################################
















