from .cn_extractor import CNFeat

_cn_extractor = CNFeat()


def extract(image):
    """

    :param image: ndarray, dtype=np.uint8 bgr
    :return: color name features, with channel num 10
    """
    return _cn_extractor.extract(image)

__all__ = ["extract"]