import numpy as np
import matplotlib.pyplot as plt

import numba

import cv2
from PIL import Image
import os
import tqdm

def filterBackground(img, tolerance=.05):
    backgroundColor = np.mean([np.median(img[i,:], axis=0) for i in range(img.shape[0])], axis=0)
    editedImg = img

    editedImg[np.sqrt(np.sum((img - backgroundColor)**2, axis=-1)) < tolerance*np.max(img),:] = 0
    return editedImg


def equalizeSpatialGradients(img, strength=.5, debug=False):
    horizontalGrad = np.array([np.median(img[:,i,:], axis=0) for i in range(img.shape[1])])
    verticalGrad = np.array([np.median(img[i,:,:], axis=0) for i in range(img.shape[0])])

    horizontalCorr = horizontalGrad - np.mean(horizontalGrad, axis=0)
    verticalCorr = verticalGrad - np.mean(verticalGrad, axis=0)

    corr = np.transpose([np.add.outer(horizontalCorr[:,i].T, verticalCorr[:,i].T) for i in range(3)])

    corrImg = img - corr*strength
    corrImg = (corrImg - np.min(corrImg))
    corrImg /= np.max(corrImg)

    # Now make sure the image has the same type and min/max as the input
    if img.dtype == np.uint8:
        corrImg *= 255
        corrImg = corrImg.astype(np.uint8)

    if debug:
        correctedHorizontalGrad = np.array([np.median(corrImg[:,i,:], axis=0) for i in range(img.shape[1])])
        correctedVerticalGrad = np.array([np.median(corrImg[i,:,:], axis=0) for i in range(img.shape[0])])

        fig, ax = plt.subplots(1, 2)
        labels = ['Red', 'Green', 'Blue']
        for i in range(3):
            ax[0].plot(horizontalGrad[:,i], label=labels[i], c=labels[i])
            ax[0].plot(correctedHorizontalGrad[:,i], label=f'Corr. {labels[i]}', c=f'tab:{labels[i]}')

        ax[0].set_title('Horizontal')
        ax[0].legend()
        for i in range(3):
            ax[1].plot(verticalGrad[:,i], label=labels[i], c=labels[i])
            ax[1].plot(correctedVerticalGrad[:,i], label=f'Corr. {labels[i]}', c=f'tab:{labels[i]}')

        ax[1].set_title('Vertical')
        ax[1].legend()
        fig.tight_layout()
        plt.show()

    return corrImg

def checkImage(img):
    if isinstance(img, str):
        # I don't want to overwrite the image itself, so create a new var for that
        newFrame = np.array(cv2.imread(img), dtype=np.uint8)
    else:
        newFrame = img

    return newFrame


def cropToContent(img, returnCorner=True):
    gray = np.mean(img, axis=-1)
    verticalBins = np.where(np.sum(gray, axis=0) > 0)
    if len(verticalBins[0]) > 0:
        leftBound, rightBound = verticalBins[0][0], verticalBins[0][-1]
    else:
        raise img

    horizontalBins = np.where(np.sum(gray, axis=1) > 0)
    if len(horizontalBins[0]) > 0:
        topBound, bottomBound = horizontalBins[0][0], horizontalBins[0][-1]
    else:
        raise img

    croppedImg = np.array(img)[topBound:bottomBound, leftBound:rightBound]
    return (croppedImg, (leftBound, topBound)) if returnCorner else croppedImg


DETECTOR_OPTIONS = ['sift',
                    'fast',
                    'orb',
                    'akaze']

def detectFeatures(images, detectorType=DETECTOR_OPTIONS[0], bar=True, **kwargs):
    """

    """

    # Load images
    imgArr = [checkImage(img) for img in images]

    # Initialize detector
    if detectorType == 'sift': 
        # SIFT
        detector = cv2.SIFT_create()
        detectFunc = detector.detectAndCompute

    elif detectorType == 'fast':
        # FAST
        detector = cv2.FastFeatureDetector_create()
        detector.setNonmaxSuppression(False)
        detectFunc = detector.detect

    elif detectorType == 'orb':
        # ORB
        detector = cv2.ORB_create(nfeatures=1000)
        detectFunc = detector.detectAndCompute

    elif detectorType == 'akaze':
        # AKAZE
        detector = cv2.AKAZE_create()
        detectFunc = detector.detectAndCompute

    features = [] 
    for i in tqdm.tqdm(range(len(imgArr)), desc='Feature detection') if bar else range(len(imgArr)):
        kp, des = detectFunc(np.mean(imgArr[i], axis=-1).astype(np.uint8), None, **kwargs)
        features.append({"keypoints": kp,
                         "descriptors": des})

    return features

def computeAlignment(features1, features2, matcherType='flann', ratioThreshold=.5):
    if matcherType == 'flann':
        # FLANN
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    else:
        # Brute force
        matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(features1["descriptors"], features2["descriptors"], k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < ratioThreshold*n.distance:
            good.append(m)

    da = []
    dr = []

    for match in good:
        keypoint1 = features1["keypoints"][match.queryIdx]
        keypoint2 = features2["keypoints"][match.trainIdx]

        dr.append(np.array(keypoint1.pt) - np.array(keypoint2.pt))
        da.append(keypoint1.angle - keypoint2.angle)

    return np.median(dr, axis=0), np.median(da)


BLENDER_OPTIONS = ['multiband',
                   'feather',
                   'no']

SEAM_FINDER_OPTIONS = ['color',
                       'colorgrad']

def stitchImages(imgArr, stitchCorners, stitchAngles=None, extraPadding=300, seamFinderType=SEAM_FINDER_OPTIONS[0], blenderType=BLENDER_OPTIONS[0], blenderStrength=1, crop=True):

    left, right = np.min(stitchCorners[:,1]), np.max(stitchCorners[:,1])
    top, bottom = np.min(stitchCorners[:,0]), np.max(stitchCorners[:,0])

    approxStitchSize = (int(np.abs(right - left) + imgArr[0].shape[1] + extraPadding),
                        int(np.abs(bottom - top) + imgArr[-1].shape[0] + extraPadding))

    stitchedImage = Image.new('RGB', approxStitchSize)

    corners = [[0, 0]] + [tuple(arr) for arr in stitchCorners.astype(np.int16)]
    imgSizes = [tuple(img.shape[:2])[::-1] for img in imgArr]

    seamFinder = cv2.detail_DpSeamFinder(seamFinderType.upper())
    seamMasks = seamFinder.find(list(imgArr), corners, [255*np.ones(img.shape[:2], dtype=np.uint8) for img in imgArr])

    dst_sz = cv2.detail.resultRoi(corners=corners, sizes=imgSizes)
    blendWidth = np.sqrt(dst_sz[2] * dst_sz[3]) * blenderStrength / 100

    if blenderType  == "no" or blendWidth < 1:
        blender = cv2.detail.Blender_createDefault(cv.detail.Blender_NO)

    elif blenderType == "multiband":
        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(int((np.log(blendWidth) / np.log(2.0) - 1.0)))

    elif blenderType  == "feather":
        blender = cv2.detail_FeatherBlender()
        blender.setSharpness(1.0 / blendWidth)

    blender.prepare(dst_sz)

    for i in range(len(imgArr)):
        blender.feed(cv2.UMat(np.array(imgArr[i]).astype(np.int16)), seamMasks[i], corners[i])
        
    stitchedImage = None
    stitchedMask = None
    stitchedImage, stitchedMask= blender.blend(stitchedImage, stitchedMask)

    stitchedImage, cropCorner = cropToContent(stitchedImage, True)

    return stitchedImage

