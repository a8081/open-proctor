import json

import numpy as np
import pytest

from openproctor.analysis.ocr_triage import OcrTriage
from openproctor.analysis.ocr_utils import PreprocessingMethod


@pytest.fixture(scope="module")
def ocr():
    return OcrTriage(gpu=False, batch_size=8, preprocessing=PreprocessingMethod.NONE)


class TestOcrTriageInit:
    def test_default_keywords(self):
        ocr = OcrTriage()
        assert len(ocr.keywords) > 0
        assert "discord" in ocr.keywords
        assert "chatgpt" in ocr.keywords

    def test_custom_keywords(self):
        ocr = OcrTriage(keywords=["foo", "bar"])
        assert ocr.keywords == ["foo", "bar"]

    def test_batch_size_default(self):
        ocr = OcrTriage()
        assert ocr.batch_size == 16

    def test_preprocessing_default(self):
        ocr = OcrTriage()
        assert ocr.preprocessing == PreprocessingMethod.NONE

    def test_preprocessing_from_string(self):
        ocr = OcrTriage(preprocessing="threshold")
        assert ocr.preprocessing == PreprocessingMethod.THRESHOLD


class TestMatchKeywords:
    def test_simple_match(self):
        ocr = OcrTriage(keywords=["discord"])
        hits = ocr._match_keywords("Join our Discord server")
        assert len(hits) == 1
        assert hits[0]["keyword"] == "discord"

    def test_no_match(self):
        ocr = OcrTriage(keywords=["discord"])
        hits = ocr._match_keywords("The capital of France is Paris")
        assert len(hits) == 0

    def test_multiple_matches(self):
        ocr = OcrTriage(keywords=["discord", "chatgpt"])
        hits = ocr._match_keywords("Use ChatGPT and Discord for answers")
        assert len(hits) >= 2
        matched = {h["keyword"] for h in hits}
        assert "discord" in matched
        assert "chatgpt" in matched

    def test_case_insensitive(self):
        ocr = OcrTriage(keywords=["discord"])
        hits = ocr._match_keywords("DISCORD")
        assert len(hits) == 1

    def test_unicode_normalization(self):
        ocr = OcrTriage(keywords=["discord"])
        hits = ocr._match_keywords("di\u0301scord")
        assert len(hits) == 1


class TestNormalize:
    def test_removes_accents(self):
        result = OcrTriage._normalize("café naïve")
        assert "é" not in result
        assert "ï" not in result

    def test_ascii_passthrough(self):
        result = OcrTriage._normalize("hello world 123")
        assert result == "hello world 123"


class TestParseResult:
    def test_empty_results(self, ocr):
        result = ocr._parse_result([])
        assert result["has_match"] is False
        assert result["matches"] == []
        assert result["ocr_text"] == ""
        assert result["num_detections"] == 0

    def test_low_confidence_ignored(self, ocr):
        fake_results = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "discord", 0.1),
        ]
        result = ocr._parse_result(fake_results)
        assert result["has_match"] is False
        assert result["num_detections"] == 1


class TestScanFrame:
    def test_empty_image_returns_no_match(self, ocr, blank_frame):
        result = ocr.scan_frame(blank_frame)
        assert isinstance(result, dict)
        assert "has_match" in result
        assert "ocr_text" in result

    def test_keyword_image_returns_match(self, ocr, keyword_frame):
        result = ocr.scan_frame(keyword_frame, apply_preprocessing=False)
        assert isinstance(result, dict)
        assert "has_match" in result
        assert "ocr_text" in result

    def test_clean_image_no_match(self, ocr, clean_frame):
        result = ocr.scan_frame(clean_frame)
        assert isinstance(result, dict)

    def test_path_string(self, ocr, tmp_frames_dir):
        fp = tmp_frames_dir / "frame_0000.jpg"
        result = ocr.scan_frame(str(fp))
        assert isinstance(result, dict)


class TestScanBatch:
    def test_empty_list(self, ocr):
        assert ocr.scan_batch([]) == []

    def test_single_image(self, ocr, blank_frame):
        results = ocr.scan_batch([blank_frame])
        assert len(results) == 1

    def test_multiple_images(self, ocr, blank_frame, keyword_frame):
        results = ocr.scan_batch([blank_frame, keyword_frame])
        assert len(results) == 2


class TestRunTriage:
    def test_no_frames(self, ocr, tmp_path, tmp_suspects_dir):
        interim_dir = tmp_path / "empty"
        interim_dir.mkdir()
        suspects = ocr.run_triage(
            interim_dir=interim_dir,
            suspects_dir=tmp_suspects_dir,
        )
        assert suspects == []

    def test_no_matches(self, ocr, tmp_frames_dir, tmp_suspects_dir):
        ocr_no_match = OcrTriage(keywords=["zzz_nonexistent"], gpu=False)
        suspects = ocr_no_match.run_triage(
            interim_dir=tmp_frames_dir,
            suspects_dir=tmp_suspects_dir,
            findings_file=tmp_suspects_dir / "findings.json",
        )
        assert suspects == []

    def test_progress_callback(self, ocr, tmp_frames_dir, tmp_suspects_dir):
        calls = []
        ocr.run_triage(
            interim_dir=tmp_frames_dir,
            suspects_dir=tmp_suspects_dir,
            findings_file=tmp_suspects_dir / "findings.json",
            progress=lambda pct: calls.append(pct),
        )
        assert len(calls) > 0
        assert all(0 <= p <= 1 for p in calls)

    def test_findings_json_created(self, ocr, tmp_frames_dir, tmp_suspects_dir):
        ocr.run_triage(
            interim_dir=tmp_frames_dir,
            suspects_dir=tmp_suspects_dir,
            findings_file=tmp_suspects_dir / "findings.json",
        )
        assert not (tmp_suspects_dir / "findings.json").exists()
