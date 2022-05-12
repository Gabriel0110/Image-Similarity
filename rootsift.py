import numpy as np
import cv2
from imutils.feature.factories import DescriptorExtractor_create

class RootSIFT:
    def __init__(self):
        """
        Initialize the SIFT feature extractor
        """
        self.extractor = DescriptorExtractor_create("SIFT")

    def compute(self, image, kps, eps=1e-7):
        """
        Compute the SIFT descriptors
        """
        (kps, descs) = self.extractor.compute(image, kps)

        # If there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # Apply the Hellinger kernel to the descriptors
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)

        # Return a tuple of the keypoints and descriptors
        return (kps, descs)