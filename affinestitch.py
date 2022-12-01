import numpy as np
import matplotlib.pyplot as plt

import numba

import cv2
from PIL import Image
import os
import tqdm

import stitching as st

#@numba.jit(nopython=True)
def gSquared(properFrame):
    """
    The gradient squared at each pixel of the image, also known as the convolution
    of a Laplacian kernel across the image.
    Optimized via `numba`.
    Edge values are padded with values of 0.
    Parameters
    ----------
    properFrame : np.ndarray[H,W]
        An array representing a single channel of an image.
    Returns
    -------
    gSqr : np.ndarray[H,W]
        The gradient squared at every point.
    References
    ----------
    [1] DanielsLab Matlab implementation, https://github.com/DanielsNonlinearLab/Gsquared
    [2] Abed Zadeh, A., Bares, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N.,
    Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H.
    (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular
    Matter, 21(4), 83. [10.1007/s10035-019-0942-2](https://doi.org/10.1007/s10035-019-0942-2)
    """

    # Take the full size of the image, though know that the outermost row and
    # column of pixels will be 0
    gSquared = np.zeros_like(properFrame)

    # Iterate over every pixel
    # Regardless of whether we have a good region, G^2 needs a buffer of 1 pixel on each
    # side, so we have to crop down more
    for j in range(1, np.shape(properFrame)[0]-1):
        for k in range(1, np.shape(properFrame)[1]-1):
            # I've put a little picture of which pixels we are comparing
            # for each calculation (O is the current pixel, X are the
            # ones we are calculating)

            # - - -
            # X O X
            # - - -
            g1 = float(properFrame[j, k-1]) - float(properFrame[j, k+1])

            # - X -
            # - O -
            # - X -
            g2 = float(properFrame[j-1, k]) - float(properFrame[j+1, k])

            # - - X
            # - O -
            # X - -
            g3 = float(properFrame[j-1, k+1]) - float(properFrame[j+1, k-1])

            # X - -
            # - O -
            # - - X
            g4 = float(properFrame[j-1, k-1]) - float(properFrame[j+1, k+1])

            gSquared[j,k] = (g1*g1/4.0 + g2*g2/4.0 + g3*g3/8.0 + g4*g4/8.0)/4.0

    return gSquared


def affineStitch(imagePath, ):
    #imagePath = '/home/jack/Videos/map_2_images/'
    imageExtension = 'png'
    fileTemplateName = 'image'

    detectorType = 'akaze'

    images = [imagePath + i for i in os.listdir(imagePath) if imageExtension in i]
    imageIndexing = [int(img.split(fileTemplateName)[-1].split('.')[0]) for img in images]

    images = np.array(images)[np.argsort(imageIndexing)]

    stableImageIndices = st.identifyStableImages(images, derThreshold=.05)

    stitchedOffsets = [[0,0]]
    stitchedRotations = [0]

    imgArr = [cv2.imread(images[i]) for i in stableImageIndices]

    fd = st.FeatureDetector(detectorType)
    features = [fd.detect_features(np.mean(imgArr[i], axis=-1)) for i in range(len(imgArr))]

    for i in tqdm.tqdm(range(len(imgArr)-1)):

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(features[i]["descriptors"], features[i+1]["descriptors"], k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        da = []
        dr = []

        for match in good:
            keypoint1 = features[i]["keypoints"][match.queryIdx]
            keypoint2 = features[i+1]["keypoints"][match.trainIdx]

            dr.append(np.array(keypoint1.pt) - np.array(keypoint2.pt))
            da.append(keypoint1.angle - keypoint2.angle)

        # Fix periodicity
        da = np.where(np.array(da) > 180, np.array(da) - 360, da)
        da = np.where(np.array(da) < -180, np.array(da) + 360, da)

        dr = np.array(dr)
        
        stitchedOffsets.append(np.median(dr, axis=0))
        stitchedRotations.append(np.median(da))
        
    stitchedOffsets = np.array(stitchedOffsets)
    stitchedRotations = np.array(stitchedRotations)

    # concat images
    #approxStitchSize = (5*int(np.abs(np.sum(stitchedOffsets[:,0]))),
    #                    5*int(np.abs(np.sum(stitchedOffsets[:,0]))))

    approxStitchSize = (12000, 12000)

    pastedImages = []
    corners = []

    stitchedImage = Image.new('RGB', approxStitchSize)
    currOffset = np.array([5*approxStitchSize[0]/6, 5*approxStitchSize[1]/6])
    currRotation = 0

    for i in tqdm.tqdm(range(len(imgArr))):
        currOffset = currOffset + stitchedOffsets[i]
        
        # Grab what the region that is about to be pasted over looks like
        currRegionImg = np.array(stitchedImage.rotate(-currRotation, center=(currOffset[1]+imgArr[i].shape[0]/2, currOffset[0]+imgArr[i].shape[1]/2)), dtype=np.float64)[int(currOffset[1]):int(currOffset[1])+imgArr[i].shape[0],int(currOffset[0]):int(currOffset[0])+imgArr[i].shape[1]]
        currRegionImg[currRegionImg == 0] = np.nan
        
        oldHorizontalGrad = np.array([np.nanmedian(currRegionImg[:,j,:], axis=0) for j in range(currRegionImg.shape[1])])
        oldVerticalGrad = np.array([np.nanmedian(currRegionImg[j,:,:], axis=0) for j in range(currRegionImg.shape[0])])
        
        newHorizontalGrad = np.array([np.median(imgArr[i][:,j,:], axis=0) for j in range(imgArr[i].shape[1])])
        newVerticalGrad = np.array([np.median(imgArr[i][j,:,:], axis=0) for j in range(imgArr[i].shape[0])])

    #     plt.plot(oldHorizontalGrad, label='old')
    #     plt.plot(newHorizontalGrad, label='new')
    #     plt.legend()
    #     plt.show()
        
    #     plt.plot(oldVerticalGrad, label='old')
    #     plt.plot(newVerticalGrad, label='new')
    #     plt.legend()
    #     plt.show()
        
        #img = np.where(1 - np.isnan(currRegionImg), imgArr[i]*.5 + currRegionImg*.5, imgArr[i])
        #img = st.equalizeSpatialGradients(img, strength=.1)
        img = imgArr[i]
        
        img = Image.fromarray(img.astype(np.uint8))
        img.rotate(currRotation + stitchedRotations[i])

        pastedImages.append(img)
        corners.append((int(currOffset[0]), int(currOffset[1])))
        # plt.imshow(currRegionImg.astype(np.uint8))
        # plt.show()
        
        stitchedImage.paste(img, (int(currOffset[0]), int(currOffset[1])))
        currRotation += stitchedRotations[i]
        
    stitchedImage, cropCorner = st.cropToContent(stitchedImage, True)
    corners = np.array(corners) - np.array(cropCorner)
    
    Image.fromarray(stitchedImage).save('stitch_result.png')

if __name__ == '__main__':
    imagePath = '/bucket/DaniU/Members/Jack Featherstone/Test/map_2_images/'

    affineStitch(imagePath)
