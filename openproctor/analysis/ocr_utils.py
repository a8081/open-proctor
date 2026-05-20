from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from loguru import logger


class PreprocessingMethod(str, Enum):
    NONE = "none"
    GRAYSCALE = "grayscale"
    THRESHOLD = "threshold"
    ADAPTIVE = "adaptive"
    DENOISE = "denoise"


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def preprocess_image(
    image: np.ndarray,
    method: PreprocessingMethod = PreprocessingMethod.NONE,
) -> np.ndarray:
    if method == PreprocessingMethod.NONE:
        return image

    if method == PreprocessingMethod.GRAYSCALE:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image

    if method == PreprocessingMethod.THRESHOLD:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    if method == PreprocessingMethod.ADAPTIVE:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    if method == PreprocessingMethod.DENOISE:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return image


def batch_preprocess(
    images: list[np.ndarray],
    method: PreprocessingMethod = PreprocessingMethod.NONE,
) -> list[np.ndarray]:
    if method == PreprocessingMethod.NONE:
        return images
    return [preprocess_image(img, method) for img in images]
