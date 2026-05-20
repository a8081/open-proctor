from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openproctor.video.extractor import FrameExtractor


def _make_dummy_video(path: Path):
    path.write_bytes(b"dummy mp4 content")


class TestFrameExtractorInit:
    def test_default_params(self, tmp_path):
        v = tmp_path / "dummy.mp4"
        _make_dummy_video(v)
        fe = FrameExtractor(video_path=v, output_dir=tmp_path / "out")
        assert fe.jump_sec == 5
        assert fe.similarity_pct == 95
        assert fe.pixel_tolerance == 30

    def test_custom_params(self, tmp_path):
        v = tmp_path / "dummy.mp4"
        _make_dummy_video(v)
        fe = FrameExtractor(
            video_path=v,
            output_dir=tmp_path / "out",
            jump_sec=10,
            similarity_pct=90,
            pixel_tolerance=20,
        )
        assert fe.jump_sec == 10
        assert fe.similarity_pct == 90
        assert fe.pixel_tolerance == 20


class TestSimilarityRatio:
    @pytest.fixture
    def fe(self, tmp_path):
        v = tmp_path / "dummy.mp4"
        _make_dummy_video(v)
        return FrameExtractor(video_path=v, output_dir=tmp_path / "out")

    def test_identical(self, fe):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert fe._similarity_ratio(img, img) == 1.0

    def test_completely_different(self, fe):
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        black = np.full((100, 100, 3), 0, dtype=np.uint8)
        assert fe._similarity_ratio(white, black) == 0.0

    def test_slightly_different(self, fe):
        a = np.full((50, 50, 3), 200, dtype=np.uint8)
        b = a.copy()
        b[0, 0] = (0, 0, 0)
        ratio = fe._similarity_ratio(a, b)
        assert 0.9 < ratio <= 1.0


class TestIsUnique:
    @pytest.fixture
    def fe(self, tmp_path):
        v = tmp_path / "dummy.mp4"
        _make_dummy_video(v)
        return FrameExtractor(video_path=v, output_dir=tmp_path / "out", similarity_pct=95)

    def test_identical_not_unique(self, fe):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert fe._is_unique(img, img) is False

    def test_different_is_unique(self, fe):
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        black = np.full((100, 100, 3), 0, dtype=np.uint8)
        assert fe._is_unique(white, black) is True


class TestExtractEdgeCases:
    def test_no_frames_when_cap_fails(self, tmp_path):
        v = tmp_path / "broken.mp4"
        _make_dummy_video(v)
        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = False
            mock_cap.return_value = mock_instance
            fe = FrameExtractor(video_path=v, output_dir=tmp_path / "out")
            with pytest.raises(RuntimeError, match="Cannot open video"):
                fe.extract()

    def test_zero_duration_video(self, tmp_path):
        v = tmp_path / "zero.mp4"
        _make_dummy_video(v)
        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda x: {7: 0, 5: 0, 3: 30}.get(int(x), 0)
            mock_instance.read.return_value = (False, None)
            mock_cap.return_value = mock_instance
            fe = FrameExtractor(video_path=v, output_dir=tmp_path / "out")
            with pytest.raises(ZeroDivisionError):
                fe.extract()

    def test_progress_callback(self, tmp_path):
        v = tmp_path / "test.mp4"
        _make_dummy_video(v)
        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_instance.get.side_effect = lambda x: {7: 300, 5: 30, 3: 30}.get(int(x), 0)
            mock_instance.read.return_value = (True, np.full((100, 100, 3), 128, dtype=np.uint8))
            mock_cap.return_value = mock_instance
            fe = FrameExtractor(video_path=v, output_dir=tmp_path / "out", jump_sec=5)
            calls = []
            fe.extract(progress=lambda pct: calls.append(pct))
            assert len(calls) > 0
            assert all(0 <= p <= 1 for p in calls)
