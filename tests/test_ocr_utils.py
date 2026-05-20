import numpy as np
import pytest

from openproctor.analysis.ocr_utils import (
    PreprocessingMethod,
    batch_preprocess,
    cuda_available,
    preprocess_image,
)


class TestPreprocessingMethod:
    def test_enum_values(self):
        assert PreprocessingMethod.NONE.value == "none"
        assert PreprocessingMethod.GRAYSCALE.value == "grayscale"
        assert PreprocessingMethod.THRESHOLD.value == "threshold"
        assert PreprocessingMethod.ADAPTIVE.value == "adaptive"
        assert PreprocessingMethod.DENOISE.value == "denoise"

    def test_from_string(self):
        assert PreprocessingMethod("none") == PreprocessingMethod.NONE
        assert PreprocessingMethod("threshold") == PreprocessingMethod.THRESHOLD


class TestPreprocessImage:
    def test_none_returns_original(self, blank_frame):
        result = preprocess_image(blank_frame, PreprocessingMethod.NONE)
        assert result is blank_frame

    def test_grayscale_keeps_3ch(self, blank_frame):
        result = preprocess_image(blank_frame, PreprocessingMethod.GRAYSCALE)
        assert result.shape == blank_frame.shape
        assert result.dtype == blank_frame.dtype

    def test_threshold_returns_binary(self, keyword_frame):
        result = preprocess_image(keyword_frame, PreprocessingMethod.THRESHOLD)
        assert result.shape == keyword_frame.shape
        unique = np.unique(result)
        assert len(unique) <= 2 or (unique.max() - unique.min()) < 50

    def test_adaptive_returns_3ch(self, keyword_frame):
        result = preprocess_image(keyword_frame, PreprocessingMethod.ADAPTIVE)
        assert result.shape == keyword_frame.shape
        assert result.dtype == keyword_frame.dtype

    def test_denoise_returns_same_shape(self, keyword_frame):
        result = preprocess_image(keyword_frame, PreprocessingMethod.DENOISE)
        assert result.shape == keyword_frame.shape
        assert result.dtype == keyword_frame.dtype

    def test_cuda_returns_bool(self):
        assert isinstance(cuda_available(), bool)


class TestBatchPreprocess:
    def test_none_returns_same_objects(self, blank_frame, keyword_frame):
        images = [blank_frame, keyword_frame]
        result = batch_preprocess(images, PreprocessingMethod.NONE)
        assert result[0] is blank_frame
        assert result[1] is keyword_frame

    def test_empty_list(self):
        assert batch_preprocess([]) == []

    def test_batch_threshold(self, blank_frame, keyword_frame):
        images = [blank_frame, keyword_frame]
        result = batch_preprocess(images, PreprocessingMethod.THRESHOLD)
        assert len(result) == 2
        assert result[0].shape == blank_frame.shape
        assert result[1].shape == keyword_frame.shape


class TestCudaAvailable:
    def test_returns_bool(self):
        result = cuda_available()
        assert isinstance(result, bool)
