import numpy as np
from PIL import Image

import affinestitcher as af
import os
import argparse

import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument(dest='inputPath', type=str, help="Path to directory containing image files or video")

    parser.add_argument('-d', type=str, default='sift', dest='detectorType', help='Detector type')
    parser.add_argument('-m', type=str, default='flann', dest='matcherType', help='Matcher type')
    parser.add_argument('-s', type=str, default='color', dest='seamFinderType', help='Seam finder type')
    parser.add_argument('-b', type=str, default='multiband', dest='blenderType', help='Blender type')
    parser.add_argument('-o', type=str, default='stitched_image.png', dest='outputFile', help='Output file name')

    args = parser.parse_args()

    imageExtension = 'png'
    videoExtension = 'avi'
    fileTemplateName = 'image'

    if args.inputPath.split('.')[-1] == videoExtension:
        print('Reading video frames...')
        images = af.getVideoFrames(args.inputPath)

    elif os.path.isdir(args.inputPath):
        images = [args.inputPath + '/' + i for i in os.listdir(args.inputPath) if imageExtension in i]
        imageIndexing = [int(img.split(fileTemplateName)[-1].split('.')[0]) for img in images]

        images = np.array(images)[np.argsort(imageIndexing)]
    else:
        raise Exception('Invalid input file(s) given')

    stableImageIndices = af.identifyStableImages(images, derThreshold=.05, debug=False)
    print(f'Found {len(stableImageIndices)} stable images')

    print('Loading images...')
    imgArr = [cv2.imread(images[i]) for i in stableImageIndices]

    features = af.detectFeatures(imgArr, args.detectorType)

    stitchOffsetArr = []
    stitchAngleArr = []

    print('Computing alignment...')
    for i in range(len(imgArr)-1):
        dr, da = af.computeAlignment(features[i], features[i+1], args.matcherType)
        
        stitchOffsetArr.append(dr)
        stitchAngleArr.append(da)
        
    absoluteStitchPositions = np.cumsum(stitchOffsetArr, axis=0)
        
    #corners = [tuple(arr) for arr in absoluteStitchPositions.astype(np.int16)]
    
    print('Stitching image...')
    stitchedImg = af.stitchImages(imgArr, absoluteStitchPositions, seamFinderType=args.seamFinderType, blenderType=args.blenderType)

    Image.fromarray(np.array(stitchedImg, dtype=np.uint8)).save(args.outputFile)

