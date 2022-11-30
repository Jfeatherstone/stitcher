from collections import OrderedDict

import cv2 as cv

import numpy as np

class FeatureDetector:
    """https://docs.opencv.org/4.x/d0/d13/classcv_1_1Feature2D.html"""

    DETECTOR_CHOICES = OrderedDict()

    DETECTOR_CHOICES["orb"] = 'orb'
    DETECTOR_CHOICES["sift"] = 'sift'
    DETECTOR_CHOICES["fast"] = 'fast'
    DETECTOR_CHOICES["brisk"] = cv.BRISK_create
    DETECTOR_CHOICES["akaze"] = cv.AKAZE_create

    DEFAULT_DETECTOR = list(DETECTOR_CHOICES.keys())[0]

    def __init__(self, detector=DEFAULT_DETECTOR, **kwargs):
        self.detectorType = detector

        if self.detectorType == 'sift':
            # SIFT
            self.detector = cv.SIFT_create()

        elif self.detectorType == 'fast':
            # FAST
            self.detector = cv.FastFeatureDetector_create()
            self.detector.setNonmaxSuppression(False)

        elif self.detectorType == 'orb':
            self.detector = cv.ORB_create(nfeatures=1000)

        else:
            # Other
            self.detector = FeatureDetector.DETECTOR_CHOICES[detector](**kwargs)

    def detect_features(self, img, *args, **kwargs):

        features = {}

        if self.detectorType == 'sift':
            kp, des = self.detector.detectAndCompute(np.array(img, dtype=np.uint8), None, **kwargs)
            features["keypoints"] = kp
            features["descriptors"] = des

        elif self.detectorType == 'fast':
            kp = self.detector.detect(np.array(img, dtype=np.uint8), None, **kwargs)
            features["keypoints"] = kp
            features["descriptors"] = None

        elif self.detectorType == 'orb':
            kp, des = self.detector.detectAndCompute(np.array(img, dtype=np.uint8), None)
            features["keypoints"] = kp
            features["descriptors"] = des

        else:
            f = cv.detail.computeImageFeatures2(self.detector, np.array(img, dtype=np.uint8), *args, **kwargs)
            features["keypoints"] = f.keypoints
            features["descriptors"] = f.descriptors

        return features

    @staticmethod
    def draw_keypoints(img, features, **kwargs):
        kwargs.setdefault("color", (0, 255, 0))
        return cv.drawKeypoints(img, features["keypoints"], None, **kwargs)
