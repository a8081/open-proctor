import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from loguru import logger

from openproctor.analysis.ocr_triage import OcrTriage
from openproctor.video.extractor import FrameExtractor
from openproctor.vlm.ollama_client import (
    AggregationStrategy,
    VLMConfig,
    OllamaVLM,
)


def _parse_timestamp(filename: str) -> str:
    m = re.search(r"min_(\d+)_seg_(\d+)", filename)
    if m:
        return f"{int(m.group(1))}m {int(m.group(2))}s"
    return filename


def _unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    stem = base.stem
    parent = base.parent
    for i in range(1, 999):
        candidate = parent / f"{stem}_{i}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find available directory name for {base}")


class Pipeline:
    def __init__(
        self,
        video_path: str | Path,
        interim_dir: str | Path | None = None,
        suspects_dir: str | Path | None = None,
        report_path: str | Path | None = None,
        jump_sec: int = 5,
        ocr_gpu: bool = True,
        ocr_batch_size: int = 16,
        ocr_preprocessing: str = "none",
        vlm_model: str | list[VLMConfig] = "moondream",
        vlm_strategy: str = "single",
    ):
        self.video_path = Path(video_path)
        stem = self.video_path.stem
        self.interim_dir = Path(interim_dir) if interim_dir else Path(f"data/interim/{stem}")
        self.suspects_dir = Path(suspects_dir) if suspects_dir else Path(f"data/suspects/{stem}")
        self.report_path = Path(report_path) if report_path else Path(f"data/reports/{stem}_report.json")
        self.jump_sec = jump_sec
        self.ocr_gpu = ocr_gpu
        self.ocr_batch_size = ocr_batch_size
        self.ocr_preprocessing = ocr_preprocessing
        self.vlm_model = vlm_model
        self.vlm_strategy = vlm_strategy

    def run(self, progress=None) -> dict:
        original_interim = self.interim_dir
        original_suspects = self.suspects_dir
        self.interim_dir = _unique_dir(self.interim_dir)
        self.suspects_dir = _unique_dir(self.suspects_dir)
        if self.interim_dir != original_interim:
            logger.info(f"Renaming interim dir → {self.interim_dir.name}")
        if self.suspects_dir != original_suspects:
            logger.info(f"Renaming suspects dir → {self.suspects_dir.name}")

        if progress:
            progress("extraction", 0.0, "Extrayendo frames ...")

        def _on_extract(pct):
            if progress:
                progress("extraction", pct, f"Extrayendo frames ... {pct*100:.0f}%")

        fe = FrameExtractor(
            video_path=self.video_path,
            output_dir=self.interim_dir,
            jump_sec=self.jump_sec,
        )
        stats = fe.extract(progress=_on_extract)
        n_frames = stats["saved"]
        if progress:
            progress("extraction", 1.0, f"{n_frames} frames extraídos")

        if n_frames == 0:
            return self._report({"error": "No se extrajeron frames del video"})

        if progress:
            progress("ocr", 0.0, "Analizando con OCR ...")

        def _on_ocr(pct):
            if progress:
                progress("ocr", pct, f"Analizando con OCR ... {pct*100:.0f}%")

        ocr_ok = False
        ocr_error = None
        suspects = []
        try:
            ocr = OcrTriage(
                gpu=self.ocr_gpu,
                batch_size=self.ocr_batch_size,
                preprocessing=self.ocr_preprocessing,
            )
            suspects = ocr.run_triage(
                interim_dir=self.interim_dir,
                suspects_dir=self.suspects_dir,
                progress=_on_ocr,
            )
            ocr_ok = True
        except Exception as e:
            ocr_error = str(e)
            logger.warning(f"OCR analysis failed: {ocr_error}")
            if progress:
                progress("ocr", 1.0, f"OCR no disponible — reporte basado solo en extracción")

        n_suspects = len(suspects)
        if progress:
            progress("ocr", 1.0, f"{n_suspects} sospechoso(s) detectados")

        if not ocr_ok or n_suspects == 0:
            empty = {
                "video": self.video_path.name,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_frames_extracted": n_frames,
                    "suspects_found": 0,
                    "infractions_confirmed": 0,
                    "ocr_error": ocr_error,
                },
                "infractions": [],
            }
            return self._report(empty)

        if progress:
            progress("vlm", 0.0, "Confirmando con VLM (Ollama) …")

        verdicts = []
        vlm_ok = False
        try:
            vlm = OllamaVLM(
                model=self.vlm_model,
                strategy=self.vlm_strategy,
            )
            verdicts = vlm.analyse_directory(suspects_dir=self.suspects_dir)
            vlm_ok = True
            if progress:
                progress("vlm", 1.0, f"{len(verdicts)} frame(s) analizados por VLM")
        except Exception as e:
            logger.warning(f"VLM analysis failed, report will be OCR-only: {e}")
            err_msg = str(e)[:100]
            if progress:
                progress("vlm", 1.0, f"VLM: {err_msg}")

        report = self._build_report(n_frames, suspects, verdicts, vlm_ok=vlm_ok)
        return self._report(report)

    # ------------------------------------------------------------------
    # Build final report
    # ------------------------------------------------------------------
    def _build_report(
        self,
        n_frames: int,
        suspect_paths: list[Path],
        verdicts: list[dict],
        vlm_ok: bool = False,
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

            infraction_entry = {
                "file": str(sp),
                "timestamp": _parse_timestamp(name),
                "keyword": (f_item.get("keywords") or [None])[0],
                "ocr_text": (f_item.get("ocr_text") or "")[:200],
                "infraction": True,
                "reason": v.get("reason", ""),
            }

            per_model = v.get("per_model")
            if per_model:
                infraction_entry["per_model"] = per_model

            infractions.append(infraction_entry)

        report = {
            "video": self.video_path.name,
            "timestamp": datetime.now().isoformat(),
            "vlm_available": vlm_ok,
            "summary": {
                "total_frames_extracted": n_frames,
                "suspects_found": len(suspect_paths),
                "infractions_confirmed": len(infractions),
            },
            "infractions": infractions,
        }

        strategy_used = None
        models_used = set()
        for v in verdict_map.values():
            if v.get("per_model"):
                strategy_used = v.get("strategy", "single")
                for m in v["per_model"]:
                    models_used.add(m.get("model", ""))
        if strategy_used:
            report["vlm_strategy"] = strategy_used
            report["vlm_models"] = sorted(models_used)

        return report

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
