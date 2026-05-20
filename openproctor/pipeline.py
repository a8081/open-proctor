import json
import re
from datetime import datetime
from pathlib import Path

from loguru import logger

from openproctor.analysis.ocr_triage import OcrTriage
from openproctor.video.extractor import FrameExtractor
from openproctor.vlm.ollama_client import OllamaVLM


def _parse_timestamp(filename: str) -> str:
    m = re.search(r"min_(\d+)_seg_(\d+)", filename)
    if m:
        return f"{int(m.group(1))}m {int(m.group(2))}s"
    return filename


class Pipeline:
    def __init__(
        self,
        video_path: str | Path,
        interim_dir: str | Path = "data/interim",
        suspects_dir: str | Path = "data/suspects",
        report_path: str | Path = "data/reports/report.json",
        jump_sec: int = 5,
        ocr_gpu: bool = True,
        vlm_model: str = "moondream",
    ):
        self.video_path = Path(video_path)
        self.interim_dir = Path(interim_dir)
        self.suspects_dir = Path(suspects_dir)
        self.report_path = Path(report_path)
        self.jump_sec = jump_sec
        self.ocr_gpu = ocr_gpu
        self.vlm_model = vlm_model

    def run(self, progress=None) -> dict:
        if progress:
            progress("extraction", 0.0, "Extrayendo frames ...")

        fe = FrameExtractor(
            video_path=self.video_path,
            output_dir=self.interim_dir,
            jump_sec=self.jump_sec,
        )
        stats = fe.extract()
        n_frames = stats["saved"]
        if progress:
            progress("extraction", 1.0, f"{n_frames} frames extraídos")

        if n_frames == 0:
            return self._report({"error": "No se extrajeron frames del video"})

        if progress:
            progress("ocr", 0.0, "Analizando con OCR ...")

        ocr = OcrTriage(gpu=self.ocr_gpu)
        suspects = ocr.run_triage(
            interim_dir=self.interim_dir,
            suspects_dir=self.suspects_dir,
        )
        n_suspects = len(suspects)
        if progress:
            progress("ocr", 1.0, f"{n_suspects} sospechoso(s) detectados")

        if n_suspects == 0:
            empty = {
                "video": self.video_path.name,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_frames_extracted": n_frames,
                    "suspects_found": 0,
                    "infractions_confirmed": 0,
                },
                "infractions": [],
            }
            return self._report(empty)

        if progress:
            progress("vlm", 0.0, "Confirmando con VLM (Ollama) ...")

        vlm = OllamaVLM(model=self.vlm_model)
        verdicts = vlm.analyse_directory(suspects_dir=self.suspects_dir)
        if progress:
            progress("vlm", 1.0, f"{len(verdicts)} frame(s) analizados por VLM")

        report = self._build_report(n_frames, suspects, verdicts)
        return self._report(report)

    # ------------------------------------------------------------------
    # Build final report
    # ------------------------------------------------------------------
    def _build_report(
        self,
        n_frames: int,
        suspect_paths: list[Path],
        verdicts: list[dict],
    ) -> dict:
        verdict_map = {Path(v["file"]).name: v for v in verdicts}
        findings_path = self.suspects_dir / "findings.json"
        findings_map = {}
        if findings_path.exists():
            for f_item in json.loads(findings_path.read_text()):
                findings_map[Path(f_item["file"]).name] = f_item

        infractions = []
        for sp in suspect_paths:
            name = sp.name
            v = verdict_map.get(name, {})
            f_item = findings_map.get(name, {})

            if not v.get("infraction", False):
                continue

            infractions.append(
                {
                    "file": str(sp),
                    "timestamp": _parse_timestamp(name),
                    "keyword": (f_item.get("keywords") or [None])[0],
                    "ocr_text": (f_item.get("ocr_text") or "")[:200],
                    "infraction": True,
                    "reason": v.get("reason", ""),
                }
            )

        return {
            "video": self.video_path.name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_frames_extracted": n_frames,
                "suspects_found": len(suspect_paths),
                "infractions_confirmed": len(infractions),
            },
            "infractions": infractions,
        }

    def _report(self, data: dict) -> dict:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(f"Report saved → {self.report_path}")
        logger.info(
            f"Resumen: {data.get('summary', {}).get('total_frames_extracted', 0)} frames  |  "
            f"{data.get('summary', {}).get('suspects_found', 0)} sospechosos  |  "
            f"{data.get('summary', {}).get('infractions_confirmed', 0)} infracciones confirmadas"
        )
        return data
