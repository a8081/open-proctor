import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openproctor.pipeline import Pipeline, _parse_timestamp


class TestParseTimestamp:
    def test_standard_format(self):
        assert _parse_timestamp("min_5_seg_30.jpg") == "5m 30s"

    def test_no_match_returns_filename(self):
        assert _parse_timestamp("random.jpg") == "random.jpg"


class TestPipelineReportSaving:
    def test_report_saved_to_disk(self, tmp_path):
        (tmp_path / "interim" / "test_video").mkdir(parents=True)
        pipeline = Pipeline(
            video_path=Path("nonexistent.mp4"),
            interim_dir=tmp_path / "interim" / "test_video",
            suspects_dir=tmp_path / "suspects" / "test_video",
            report_path=tmp_path / "reports" / "test_video_report.json",
        )

        report_data = {
            "video": "test_video.mp4",
            "timestamp": "2024-01-01T00:00:00",
            "summary": {"total_frames_extracted": 50, "suspects_found": 0, "infractions_confirmed": 0},
            "infractions": [],
        }

        result = pipeline._report(report_data)
        assert result == report_data
        assert (tmp_path / "reports" / "test_video_report.json").exists()

        saved = json.loads((tmp_path / "reports" / "test_video_report.json").read_text())
        assert saved["video"] == "test_video.mp4"

    def test_empty_frames_report(self, tmp_path):
        (tmp_path / "interim" / "empty_video").mkdir(parents=True)
        pipeline = Pipeline(
            video_path=Path("empty.mp4"),
            interim_dir=tmp_path / "interim" / "empty_video",
            suspects_dir=tmp_path / "suspects" / "empty_video",
            report_path=tmp_path / "reports" / "empty_video_report.json",
        )

        report = pipeline._report({"error": "No se extrajeron frames del video"})
        assert report["error"] == "No se extrajeron frames del video"


class TestPipelineEdgeCases:
    def test_empty_frames_returns_early(self, tmp_path):
        with patch("openproctor.pipeline.FrameExtractor") as mock_fe:
            mock_fe_instance = MagicMock()
            mock_fe_instance.extract.return_value = {"saved": 0}
            mock_fe.return_value = mock_fe_instance

            pipeline = Pipeline(
                video_path=Path("empty.mp4"),
                interim_dir=tmp_path / "interim" / "empty_video",
                suspects_dir=tmp_path / "suspects" / "empty_video",
                report_path=tmp_path / "reports" / "empty_video_report.json",
            )

            report = pipeline.run()
            assert "error" in report

    def test_no_suspects_returns_empty_infractions(self, tmp_path):
        with (
            patch("openproctor.pipeline.FrameExtractor") as mock_fe,
            patch("openproctor.pipeline.OcrTriage") as mock_ocr,
        ):
            mock_fe_instance = MagicMock()
            mock_fe_instance.extract.return_value = {"saved": 50}
            mock_fe.return_value = mock_fe_instance

            mock_ocr_instance = MagicMock()
            mock_ocr_instance.run_triage.return_value = []
            mock_ocr.return_value = mock_ocr_instance

            (tmp_path / "interim" / "clean_video").mkdir(parents=True)
            pipeline = Pipeline(
                video_path=Path("clean.mp4"),
                interim_dir=tmp_path / "interim" / "clean_video",
                suspects_dir=tmp_path / "suspects" / "clean_video",
                report_path=tmp_path / "reports" / "clean_video_report.json",
            )

            report = pipeline.run()
            assert report["summary"]["suspects_found"] == 0
            assert report["summary"]["infractions_confirmed"] == 0
            assert report["infractions"] == []

    def test_vlm_failure_falls_back(self, tmp_path):
        with (
            patch("openproctor.pipeline.FrameExtractor") as mock_fe,
            patch("openproctor.pipeline.OcrTriage") as mock_ocr,
            patch("openproctor.pipeline.OllamaVLM") as mock_vlm,
        ):
            mock_fe_instance = MagicMock()
            mock_fe_instance.extract.return_value = {"saved": 50}
            mock_fe.return_value = mock_fe_instance

            mock_ocr_instance = MagicMock()
            suspect_path = tmp_path / "suspects" / "suspect.jpg"
            mock_ocr_instance.run_triage.return_value = [suspect_path]
            mock_ocr.return_value = mock_ocr_instance

            mock_vlm_instance = MagicMock()
            mock_vlm_instance.analyse_directory.side_effect = Exception("VLM unavailable")
            mock_vlm.return_value = mock_vlm_instance

            (tmp_path / "suspects").mkdir(parents=True)
            (tmp_path / "suspects" / "findings.json").write_text(
                json.dumps([{"file": str(suspect_path), "keywords": ["discord"], "ocr_text": "Join discord"}])
            )

            pipeline = Pipeline(
                video_path=Path("test.mp4"),
                interim_dir=tmp_path / "interim" / "test_video",
                suspects_dir=tmp_path / "suspects",
                report_path=tmp_path / "reports" / "test_video_report.json",
            )

            report = pipeline.run()
            assert report["vlm_available"] is False
            assert report["summary"]["suspects_found"] == 1

    def test_progress_callback_receives_phases(self, tmp_path):
        with (
            patch("openproctor.pipeline.FrameExtractor") as mock_fe,
            patch("openproctor.pipeline.OcrTriage") as mock_ocr,
        ):
            mock_fe_instance = MagicMock()
            mock_fe_instance.extract.return_value = {"saved": 0}
            mock_fe.return_value = mock_fe_instance

            pipeline = Pipeline(
                video_path=Path("test.mp4"),
                interim_dir=tmp_path / "interim" / "test_video",
                suspects_dir=tmp_path / "suspects" / "test_video",
                report_path=tmp_path / "reports" / "test_video_report.json",
            )

            cb = MagicMock()
            pipeline.run(progress=cb)
            cb.assert_any_call("extraction", 0.0, "Extrayendo frames ...")
