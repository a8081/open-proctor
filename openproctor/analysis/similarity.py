from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

try:
    import imagehash
    from PIL import Image
except ImportError:
    imagehash = None
    Image = None


class SimilarityMethod(str, Enum):
    MSE = "mse"
    PHASH = "phash"
    SSIM = "ssim"


class SimilarityFilter:
    def __init__(
        self,
        method: SimilarityMethod | str = SimilarityMethod.MSE,
        threshold: float = 95.0,
        pixel_tolerance: int = 30,
    ):
        self.method = SimilarityMethod(method)
        self.threshold = threshold
        self.pixel_tolerance = pixel_tolerance

    # ------------------------------------------------------------------
    # Internal comparison helpers
    # ------------------------------------------------------------------
    def _to_gray_resized(self, a: np.ndarray, b: np.ndarray):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        h = min(ga.shape[0], gb.shape[0])
        w = min(ga.shape[1], gb.shape[1])
        return cv2.resize(ga, (w, h)), cv2.resize(gb, (w, h))

    def _compare_mse(self, a: np.ndarray, b: np.ndarray) -> float:
        ga, gb = self._to_gray_resized(a, b)
        diff = cv2.absdiff(ga, gb)
        similar = np.sum(diff <= self.pixel_tolerance)
        return float(similar) / float(ga.size) * 100.0

    def _compare_phash(self, a: np.ndarray, b: np.ndarray) -> float:
        if imagehash is None:
            raise ImportError("imagehash / Pillow not available")
        pa = imagehash.phash(Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)))
        pb = imagehash.phash(Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)))
        hamming = pa - pb
        max_dist = 64.0
        return (1.0 - hamming / max_dist) * 100.0

    def _compare_ssim(self, a: np.ndarray, b: np.ndarray) -> float:
        if ssim is None:
            raise ImportError("scikit-image not available")
        ga, gb = self._to_gray_resized(a, b)
        score, _ = ssim(ga, gb, full=True, data_range=ga.max() - ga.min())
        return score * 100.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compare(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        dispatch = {
            SimilarityMethod.MSE: self._compare_mse,
            SimilarityMethod.PHASH: self._compare_phash,
            SimilarityMethod.SSIM: self._compare_ssim,
        }
        return dispatch[self.method](img_a, img_b)

    def is_similar(self, img_a: np.ndarray, img_b: np.ndarray) -> bool:
        return self.compare(img_a, img_b) >= self.threshold

    def filter_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
    ) -> dict:
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = sorted(input_dir.iterdir())
        if not paths:
            logger.warning(f"No files in {input_dir}")
            return {"input_files": 0, "kept": 0, "removed_similar": 0}

        prev = None
        kept = 0
        removed = 0

        for p in paths:
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                continue
            frame = cv2.imread(str(p))
            if frame is None:
                continue

            if prev is not None and self.is_similar(prev, frame):
                removed += 1
                continue

            cv2.imwrite(str(output_dir / p.name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            kept += 1
            prev = frame

        logger.info(
            f"[{self.method.value}] Kept: {kept}  |  Removed (similar): {removed}"
        )
        return {"input_files": len(paths), "kept": kept, "removed_similar": removed}
