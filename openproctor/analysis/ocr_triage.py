import json
import re
import shutil
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

DEFAULT_KEYWORDS = [
    "discord",
    "chatgpt",
    "copilot",
    "gemini",
    "openai",
    "whatsapp",
    "telegram",
    "gchat",
]


class OcrTriage:
    def __init__(
        self,
        keywords: list[str] | None = None,
        lang: list[str] | None = None,
        gpu: bool = True,
        min_confidence: float = 0.3,
    ):
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.lang = lang or ["en", "es"]
        self.gpu = gpu
        self.min_confidence = min_confidence

        self._patterns = {
            kw: re.compile(re.escape(kw), re.IGNORECASE) for kw in self.keywords
        }
        self._reader = None

    @cached_property
    def reader(self):
        import easyocr

        logger.info(f"Loading EasyOCR reader (lang={self.lang}, gpu={self.gpu}) ...")
        return easyocr.Reader(self.lang, gpu=self.gpu)

    # ------------------------------------------------------------------
    # Text normalisation & keyword matching
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        import unicodedata

        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def _match_keywords(self, text: str) -> list[dict]:
        normalized = self._normalize(text)
        hits = []
        for kw, pattern in self._patterns.items():
            for m in pattern.finditer(normalized):
                hits.append({"keyword": kw, "start": m.start(), "end": m.end()})
        return hits

    # ------------------------------------------------------------------
    # Scan a single frame
    # ------------------------------------------------------------------
    def scan_frame(self, image: str | Path | np.ndarray) -> dict:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Cannot read image: {image}")

        results = self.reader.readtext(image)
        frame_matches = []
        all_text = []

        for bbox, text, conf in results:
            all_text.append(text)
            if conf < self.min_confidence:
                continue
            hits = self._match_keywords(text)
            if hits:
                frame_matches.append(
                    {"text": text, "confidence": conf, "bbox": bbox, "matches": hits}
                )

        return {
            "has_match": len(frame_matches) > 0,
            "matches": frame_matches,
            "ocr_text": " ".join(all_text),
            "num_detections": len(results),
        }

    # ------------------------------------------------------------------
    # Full triage pipeline
    #   input:   data/interim/            (frames from extractor)
    #   output:  data/suspects/           (moved frames)
    #            data/suspects/findings.json
    # ------------------------------------------------------------------
    def run_triage(
        self,
        interim_dir: str | Path = "data/interim",
        suspects_dir: str | Path = "data/suspects",
        findings_file: str | Path = "data/suspects/findings.json",
    ) -> list[Path]:
        interim_dir = Path(interim_dir)
        suspects_dir = Path(suspects_dir)
        findings_file = Path(findings_file)

        suspects_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(
            p for p in interim_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

        if not frames:
            logger.warning(f"No frames found in {interim_dir}")
            return []

        logger.info(f"OCR triage: scanning {len(frames)} frames from {interim_dir} ...")

        findings = []
        suspect_paths = []

        for fp in frames:
            result = self.scan_frame(fp)
            if not result["has_match"]:
                continue

            # Move frame to suspects
            dst = suspects_dir / fp.name
            shutil.move(str(fp), str(dst))
            suspect_paths.append(dst)

            # Collect matched keywords for this frame
            matched_keywords = list(
                {h["keyword"] for m in result["matches"] for h in m["matches"]}
            )
            findings.append(
                {
                    "file": str(dst),
                    "keywords": matched_keywords,
                    "ocr_text": result["ocr_text"],
                }
            )

            logger.info(f"  SUSPECT: {fp.name} -> {matched_keywords}")

        # Write findings JSON
        if findings:
            with open(findings_file, "w") as f:
                json.dump(findings, f, indent=2)
            logger.info(f"{len(findings)} suspect(s) logged to {findings_file}")
        else:
            logger.info("No suspects found.")

        return suspect_paths

    # ------------------------------------------------------------------
    # Batch scan (lower-level, returns results without moving)
    # ------------------------------------------------------------------
    def scan_batch(self, images: list[str | Path | np.ndarray]) -> list[dict]:
        loaded = []
        for img in images:
            if isinstance(img, (str, Path)):
                arr = cv2.imread(str(img))
                if arr is None:
                    logger.warning(f"Skipping unreadable image: {img}")
                    continue
                loaded.append(arr)
            else:
                loaded.append(img)

        if not loaded:
            return []

        outputs = []
        for arr in loaded:
            single_results = self.reader.readtext(arr)
            frame_matches = []
            all_text = []
            for bbox, text, conf in single_results:
                all_text.append(text)
                if conf < self.min_confidence:
                    continue
                hits = self._match_keywords(text)
                if hits:
                    frame_matches.append(
                        {"text": text, "confidence": conf, "bbox": bbox, "matches": hits}
                    )
            outputs.append(
                {
                    "has_match": len(frame_matches) > 0,
                    "matches": frame_matches,
                    "ocr_text": " ".join(all_text),
                    "num_detections": len(single_results),
                }
            )
        return outputs
