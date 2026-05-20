from __future__ import annotations

import json
import re
import shutil
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from openproctor.analysis.ocr_utils import (
    PreprocessingMethod,
    batch_preprocess,
    cuda_available,
    preprocess_image,
)

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
        batch_size: int = 16,
        preprocessing: str | PreprocessingMethod = PreprocessingMethod.NONE,
    ):
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.lang = lang or ["en", "es"]
        self.gpu = gpu
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.preprocessing = (
            PreprocessingMethod(preprocessing)
            if isinstance(preprocessing, str)
            else preprocessing
        )

        if gpu and not cuda_available():
            logger.warning(
                "GPU requested for EasyOCR but CUDA is not available. "
                "Falling back to CPU (will be slow). "
                "Install PyTorch with CUDA: https://pytorch.org/get-started/locally/"
            )

        self._patterns = {
            kw: re.compile(re.escape(kw), re.IGNORECASE) for kw in self.keywords
        }
        self._reader = None

    @cached_property
    def reader(self):
        import easyocr

        logger.info(
            f"Loading EasyOCR reader (lang={self.lang}, "
            f"gpu={self.gpu}, batch_size={self.batch_size}, "
            f"preprocessing={self.preprocessing.value}) ..."
        )
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
    # Parse a single EasyOCR result list -> internal dict format
    # ------------------------------------------------------------------
    def _parse_result(self, single_results: list) -> dict:
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
        return {
            "has_match": len(frame_matches) > 0,
            "matches": frame_matches,
            "ocr_text": " ".join(all_text),
            "num_detections": len(single_results),
        }

    # ------------------------------------------------------------------
    # Batch OCR scan (GPU-efficient via readtext_batched)
    # ------------------------------------------------------------------
    def _batch_scan(self, images: list[np.ndarray]) -> list[dict]:
        if not images:
            return []

        preprocessed = batch_preprocess(images, self.preprocessing)

        try:
            batch_results = self.reader.readtext_batched(
                preprocessed, batch_size=self.batch_size
            )
        except (AttributeError, TypeError):
            batch_results = self.reader.readtext(preprocessed)

        return [self._parse_result(r) for r in batch_results]

    # ------------------------------------------------------------------
    # Scan a single frame (backward-compatible)
    # ------------------------------------------------------------------
    def scan_frame(
        self,
        image: str | Path | np.ndarray,
        apply_preprocessing: bool = False,
    ) -> dict:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Cannot read image: {image}")

        if apply_preprocessing and self.preprocessing != PreprocessingMethod.NONE:
            image = preprocess_image(image, self.preprocessing)

        results = self.reader.readtext(image)
        return self._parse_result(results)

    # ------------------------------------------------------------------
    # Full triage pipeline with batch processing
    #         input:   data/interim/            (frames from extractor)
    #         output:  data/suspects/           (moved frames)
    #                  data/suspects/findings.json
    # ------------------------------------------------------------------
    def run_triage(
        self,
        interim_dir: str | Path = "data/interim",
        suspects_dir: str | Path = "data/suspects",
        findings_file: str | Path = "data/suspects/findings.json",
        progress=None,
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

        n_total = len(frames)
        logger.info(
            f"OCR triage: scanning {n_total} frames from {interim_dir} "
            f"(batch_size={self.batch_size}, preprocessing={self.preprocessing.value}) ..."
        )

        findings = []
        suspect_paths = []
        processed = 0

        batch_size = min(self.batch_size, 16)

        for batch_start in range(0, n_total, batch_size):
            batch_files = frames[batch_start : batch_start + batch_size]

            images = []
            valid_files = []
            for fp in batch_files:
                img = cv2.imread(str(fp))
                if img is None:
                    logger.warning(f"Skipping unreadable frame: {fp}")
                    continue
                images.append(img)
                valid_files.append(fp)

            if not images:
                processed += len(batch_files)
                continue

            batch_results = self._batch_scan(images)

            for i, result in enumerate(batch_results):
                processed += 1
                if progress:
                    progress(processed / n_total)

                if not result["has_match"]:
                    continue

                fp = valid_files[i]
                dst = suspects_dir / fp.name
                shutil.move(str(fp), str(dst))
                suspect_paths.append(dst)

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

        if findings:
            with open(findings_file, "w") as f:
                json.dump(findings, f, indent=2)
            logger.info(f"{len(findings)} suspect(s) logged to {findings_file}")
        else:
            logger.info("No suspects found.")

        return suspect_paths

    # ------------------------------------------------------------------
    # Batch scan (lower-level, returns results without moving files)
    # ------------------------------------------------------------------
    def scan_batch(
        self,
        images: list[str | Path | np.ndarray],
    ) -> list[dict]:
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

        return self._batch_scan(loaded)
