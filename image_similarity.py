import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import convolve
from scipy import spatial
from skimage.metrics import structural_similarity as ssim
import PIL
from PIL import Image
import imagehash

IMG_SIZE = (512, 512)
COSINE_THRESHOLD = 0.8
L2_NA_THRESHOLD = 1.0
SSIM_THRESHOLD = 0.8

def neighbor_avg(img):
    """
    Compute the average of the neighboring pixels of each pixel in the image using 3x3 reference kernel.
    """
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0

    # Convolve2d is faster than scipy.ndimage.generic_filter, especially for large images.
    neighbor_sum = convolve(img, kernel, mode='constant')
    num_neighbor = convolve(np.ones(img.shape), kernel, mode='constant')
    #neighbor_sum = convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
    #num_neighbor = convolve2d(np.ones(img.shape), kernel, mode='same', boundary='fill', fillvalue=0)

    return neighbor_sum / num_neighbor

def resize_image_ndarray(img, size: tuple):
    """
    Resize an ndarray image to the given size.
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def resize_image_pil(img, size: tuple):
    """
    Resize a PIL image to the given size.
    """
    return img.resize(size, PIL.Image.ANTIALIAS)

def get_L2_norm_neighbor_avg(ref_img, target_img):
    """
    Calculate the L2 norm of the neighbor average of the reference image and the target image.
    """
    return np.linalg.norm(neighbor_avg(ref_img) - neighbor_avg(target_img))

def get_ssim(ref_img, target_img):
    """
    Calculate the SSIM of the reference image and the target image.
    """
    return ssim(ref_img, target_img, channel_axis=2, data_range=ref_img.max() - ref_img.min())

def get_cosine_similarity(ref_img, target_img):
    """
    Calculate the cosine similarity (in percent) of the reference image and the target image.
    """
    return (-1 * (spatial.distance.cosine(ref_img, target_img) - 1))

def get_image_hash(ref_img, target_img):
    """
    Calculate the image hash of the reference image and the target image.
    """
    phashvalue = imagehash.phash(ref_img) - imagehash.phash(target_img)
    avghashvalue = imagehash.average_hash(ref_img) - imagehash.average_hash(target_img)
    total_hash = phashvalue + avghashvalue
    return total_hash

def get_opencv_image(file):
    """
    Get a resized OpenCV image from the streamlit file_uploader object.
    """
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return resize_image_ndarray(img, IMG_SIZE) / 255

def get_pil_image(file):
    """
    Get a resized PIL image from the streamlit file_uploader object.
    """
    img = Image.open(file)
    return resize_image_pil(img, IMG_SIZE)