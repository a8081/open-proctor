from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class FrameExtractor:
    def __init__(
        self,
        video_path: str | Path,
        output_dir: str | Path = "data/interim",
        jump_sec: int = 5,
        similarity_pct: float = 95.0,
        pixel_tolerance: int = 30,
    ):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.jump_sec = jump_sec
        self.similarity_pct = similarity_pct
        self.pixel_tolerance = pixel_tolerance

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------
    def _similarity_ratio(self, a: np.ndarray, b: np.ndarray) -> float:
        gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

        h = min(gray_a.shape[0], gray_b.shape[0])
        w = min(gray_a.shape[1], gray_b.shape[1])
        gray_a = cv2.resize(gray_a, (w, h))
        gray_b = cv2.resize(gray_b, (w, h))

        diff = cv2.absdiff(gray_a, gray_b)
        similar_pixels = np.sum(diff <= self.pixel_tolerance)
        return float(similar_pixels) / float(gray_a.size)

    def _is_unique(self, a: np.ndarray, b: np.ndarray) -> bool:
        return self._similarity_ratio(a, b) * 100.0 < self.similarity_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self) -> dict:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        logger.info(f"Video       : {self.video_path.name}")
        logger.info(f"Duration    : {duration_sec:.1f}s  ({total_frames} frames @ {fps:.2f} fps)")
        logger.info(f"Jump every  : {self.jump_sec}s")
        logger.info(f"Output dir  : {self.output_dir}")

        prev = None
        saved = 0
        skipped_similar = 0
        skipped_unreadable = 0
        sec = 0.0

        while sec < duration_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
            ret, frame = cap.read()
            if not ret:
                skipped_unreadable += 1
                sec += self.jump_sec
                continue

            if prev is not None and not self._is_unique(prev, frame):
                skipped_similar += 1
            else:
                mins = int(sec // 60)
                segs = int(sec % 60)
                name = f"min_{mins}_seg_{segs}.jpg"
                cv2.imwrite(str(self.output_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved += 1
                prev = frame.copy()

            sec += self.jump_sec

        cap.release()

        stats = {
            "video": self.video_path.name,
            "duration_sec": duration_sec,
            "jumps_attempted": int(duration_sec // self.jump_sec) + 1,
            "saved": saved,
            "skipped_similar": skipped_similar,
            "skipped_unreadable": skipped_unreadable,
        }

        logger.info(f"Saved: {saved}  |  Skipped (similar): {skipped_similar}  |  "
                    f"Skipped (unreadable): {skipped_unreadable}")
        return stats
