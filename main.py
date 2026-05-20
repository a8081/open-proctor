#!/usr/bin/env python3
"""CLI para OpenProctor — procesamiento por lotes.

Uso:
  python main.py video.mp4
  python main.py data/input/*.mp4
  python main.py data/input/         # procesa todos los .mp4 de la carpeta
"""

import sys
from pathlib import Path

import typer
from loguru import logger

from openproctor import Pipeline

app = typer.Typer()


def _run_single(
    video: Path,
    jump_sec: int,
    model: str,
    gpu: bool,
):
    logger.info(f"{'='*60}")
    logger.info(f"Procesando: {video.name}")
    logger.info(f"{'='*60}")

    pipeline = Pipeline(
        video_path=video,
        jump_sec=jump_sec,
        ocr_gpu=gpu,
        vlm_model=model,
    )

    report = pipeline.run()
    s = report["summary"]
    logger.info(
        f"→ {s['total_frames_extracted']} frames  |  "
        f"{s['suspects_found']} sospechosos  |  "
        f"{s['infractions_confirmed']} infracciones"
    )

    for inf in report["infractions"]:
        print(f"  🛑 {inf['timestamp']}  {inf.get('keyword','')}  |  {inf['reason']}")

    return report


@app.command()
def run(
    videos_or_dir: list[Path] = typer.Argument(
        ..., help="Uno o varios .mp4, o un directorio con videos"
    ),
    jump_sec: int = typer.Option(5, "--jump", "-j", help="Salto entre frames (seg)"),
    model: str = typer.Option("moondream", "--model", "-m", help="Modelo VLM en Ollama"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Usar GPU para OCR"),
):
    """Ejecuta el pipeline sobre uno o varios .mp4 y guarda reportes individuales."""
    videos: list[Path] = []
    for arg in videos_or_dir:
        if arg.is_dir():
            videos.extend(sorted(arg.glob("*.mp4")))
        elif arg.exists():
            videos.append(arg)
        else:
            logger.error(f"Archivo no encontrado: {arg}")
            raise typer.Exit(1)

    if not videos:
        logger.error("No se encontraron archivos .mp4")
        raise typer.Exit(1)

    total = len(videos)
    for i, v in enumerate(videos, 1):
        logger.info(f"\n[{i}/{total}] Procesando {v.name} ...")
        try:
            _run_single(v, jump_sec, model, gpu)
            logger.info(f"[{i}/{total}] ✅ {v.name} completado")
        except Exception as e:
            logger.error(f"[{i}/{total}] ❌ {v.name} falló: {e}")

    logger.info(f"\nLote completado — {total} video(s) procesado(s)")


if __name__ == "__main__":
    app()
