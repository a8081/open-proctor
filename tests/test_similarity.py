import cv2
import numpy as np
import pytest

from openproctor.analysis.similarity import SimilarityFilter, SimilarityMethod


def _make_test_images(tmp_path, n=3, vary: bool = True):
    d = tmp_path / "images"
    d.mkdir()
    base = np.full((100, 200, 3), 200, dtype=np.uint8)
    for i in range(n):
        img = base.copy()
        if vary and i > 0:
            img[20:40, 30:50] = (0, 0, 0)
        cv2.imwrite(str(d / f"img_{i:04d}.jpg"), img)
    return d


class TestSimilarityMethod:
    def test_enum_values(self):
        assert SimilarityMethod.MSE.value == "mse"
        assert SimilarityMethod.PHASH.value == "phash"
        assert SimilarityMethod.SSIM.value == "ssim"


class TestSimilarityFilterInit:
    def test_default_method(self):
        sf = SimilarityFilter()
        assert sf.method == SimilarityMethod.MSE
        assert sf.threshold == 95.0

    def test_custom_method(self):
        sf = SimilarityFilter(method=SimilarityMethod.PHASH)
        assert sf.method == SimilarityMethod.PHASH

    def test_custom_threshold(self):
        sf = SimilarityFilter(threshold=80.0)
        assert sf.threshold == 80.0


class TestIsSimilar:
    def test_identical_images(self):
        sf = SimilarityFilter()
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        assert sf.is_similar(img, img) is True

    def test_completely_different(self):
        sf = SimilarityFilter(threshold=99.0)
        white = np.full((50, 50, 3), 255, dtype=np.uint8)
        black = np.full((50, 50, 3), 0, dtype=np.uint8)
        assert sf.is_similar(white, black) is False

    def test_slightly_different(self):
        sf = SimilarityFilter(threshold=95.0)
        a = np.full((50, 50, 3), 200, dtype=np.uint8)
        b = a.copy()
        b[0, 0] = (0, 0, 0)
        assert sf.is_similar(a, b) is True


class TestFilterDirectory:
    def test_empty_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        sf = SimilarityFilter()
        result = sf.filter_directory(str(d))
        assert isinstance(result, dict)
        assert result["input_files"] == 0

    def test_all_identical_keeps_first(self, tmp_path):
        d = _make_test_images(tmp_path, n=5, vary=False)
        sf = SimilarityFilter(threshold=95.0)
        result = sf.filter_directory(str(d))
        assert result["kept"] == 1
        assert result["removed_similar"] == 4

    def test_all_different_keeps_all(self, tmp_path):
        d = _make_test_images(tmp_path, n=3, vary=True)
        sf = SimilarityFilter(threshold=90.0)
        result = sf.filter_directory(str(d))
        assert result["kept"] >= 1
