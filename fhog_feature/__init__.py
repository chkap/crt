
from . import fhog_extractor

#static PyObject* extract(PyObject* py_img, int use_hog = 2, int bin_size = 4, int n_orients = 9, int soft_bin = -1, float clip = 0.2)
def extract(image, use_hog=2, bin_size=4, n_orients=9, soft_bin=1, clip=0.2):
    """
    :param image: ndarray, rgb data, dtype=np.uint8
    :param use_hog: the type of hog feature, 0:gradients 1:hog 2:fhog
    :param bin_size: the size of cell
    :param n_orients: the number of orients
    :param soft_bin:
    :param clip: the value at which the histogram is clipped
    :return: ndarray, with channle 3*n_orients+4
    """
    feat = fhog_extractor.extract(image, use_hog, bin_size, n_orients, soft_bin, clip)
    return feat

__all__ = ["extract",]