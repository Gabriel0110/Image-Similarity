from rootsift import RootSIFT
import cv2
import numpy as np
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create

class RootSiftDriver:
    def __init__(self):
        """
        Initialize the driver for the RootSIFT process.
        """
        self.detector = FeatureDetector_create("SIFT")
        self.rs = RootSIFT()
        self.matcher = cv2.BFMatcher()
        self.img1 = None
        self.img2 = None
        self.gray_img1 = None
        self.gray_img2 = None
        self.feature_matches_img = None

    def run_sift(self, img1, img2):
        """
        Run the SIFT algorithm on the given images.
        """

        # Get the images we are going to extract descriptors from and convert to grayscale
        self.img1 = img1
        self.img2 = img2
        self.gray_img1 = cv2.cvtColor(np.float32(self.img1), cv2.COLOR_BGR2GRAY)
        self.gray_img2 = cv2.cvtColor(np.float32(self.img2), cv2.COLOR_BGR2GRAY)
        self.gray_img1 = cv2.normalize(self.gray_img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        self.gray_img2 = cv2.normalize(self.gray_img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # Detect Difference of Gaussian (DoG) keypoints in the images
        kps1 = self.detector.detect(self.gray_img1)
        kps2 = self.detector.detect(self.gray_img2)

        # Extract RootSIFT descriptors
        (kps1, descs1) = self.rs.compute(self.gray_img1, kps1)
        (kps2, descs2) = self.rs.compute(self.gray_img2, kps2)

        # Draw the keypoints on the images
        kp_img1 = cv2.drawKeypoints(self.gray_img1, kps1, self.img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kp_img2 = cv2.drawKeypoints(self.gray_img2, kps2, self.img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Feature matching
        # Using BFMatcher (brute force) here, as it going to be more accurate. FlannBasedMatcher is a much faster,
        # nearest neighbor (approximate) matcher that uses the KNN algorithm, but can be less accurate. FBM builds a KD-Tree,
        # which is an efficient data structure for searching nearest neighbors. FBM is great with large data sets (not my use case here).
        # Also going to use the knnMatch method with k=2
        matches = self.matcher.knnMatch(descs1, descs2, k=2)

        # Apply Lowe's ratio test to filter out low quality matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Determine the max number of matches that can be found based on the image with the most amount of keypoints
        max_matches = max(len(kps1), len(kps2))

        # Draw only the good matches onto an image with both images side-by-side
        #self.feature_matches_img = cv2.drawMatches(kp_img1, kps1, kp_img2, kps2, matches[:min_matches], kp_img2, flags=2)
        self.feature_matches_img = cv2.drawMatches(kp_img1, kps1, kp_img2, kps2, good_matches, kp_img2, flags=2)

        return matches, max_matches, good_matches, kp_img1, kp_img2, self.feature_matches_img

    def load_images(self, img1, img2):
        """
        Load the images by decoding the bytearray, resizing to 512x512, and converting to grayscale.
        """
        self.img1 = cv2.resize(cv2.imdecode(np.asarray(bytearray(img1.read()), dtype=np.uint8), cv2.IMREAD_COLOR), (512, 512), interpolation=cv2.INTER_AREA)
        self.img2 = cv2.resize(cv2.imdecode(np.asarray(bytearray(img2.read()), dtype=np.uint8), cv2.IMREAD_COLOR), (512, 512), interpolation=cv2.INTER_AREA)
        self.gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.gray_img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)