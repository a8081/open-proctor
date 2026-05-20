from pathlib import Path

import typer
from loguru import logger

from openproctor import Pipeline
from openproctor.vlm.ollama_client import VLMConfig, DEFAULT_MODELS

app = typer.Typer()


def _resolve_models(model_names: list[str]) -> list[VLMConfig]:
    config_map = {c.name.lower(): c for c in DEFAULT_MODELS}
    resolved = []
    for name in model_names:
        key = name.lower()
        if key in config_map:
            resolved.append(config_map[key])
        else:
            logger.warning(f"Unknown model '{name}', using default config")
            resolved.append(VLMConfig(name=name, model=name))
    return resolved


def _run_single(
    video: Path,
    jump_sec: int,
    models: list[VLMConfig],
    strategy: str,
    gpu: bool,
    batch_size: int = 16,
    preprocessing: str = "none",
):
    logger.info(f"{'='*60}")
    logger.info(f"Processing: {video.name}")
    logger.info(f"{'='*60}")

    pipeline = Pipeline(
        video_path=video,
        jump_sec=jump_sec,
        ocr_gpu=gpu,
        ocr_batch_size=batch_size,
        ocr_preprocessing=preprocessing,
        vlm_model=models,
        vlm_strategy=strategy,
    )

    report = pipeline.run()
    s = report["summary"]
    logger.info(
        f"\u2192 {s['total_frames_extracted']} frames  |  "
        f"{s['suspects_found']} suspects  |  "
        f"{s['infractions_confirmed']} infractions"
    )

    for inf in report["infractions"]:
        print(f"  \U0001f6ab {inf['timestamp']}  {inf.get('keyword','')}  |  {inf['reason']}")

    return report


@app.command()
def run(
    videos_or_dir: list[Path] = typer.Argument(
        ..., help="One or more .mp4 files, or a directory containing videos"
    ),
    jump_sec: int = typer.Option(5, "--jump", "-j", help="Seconds between extracted frames"),
    model: list[str] = typer.Option(
        ["moondream"], "--model", "-m", help="VLM model(s) in Ollama (repeatable: -m moondream -m llava)"
    ),
    strategy: str = typer.Option(
        "single", "--strategy", "-s", help="Aggregation: single, majority, consensus, any"
    ),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use GPU for OCR"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="OCR batch size"),
    preprocessing: str = typer.Option(
        "none", "--preprocess", "-p",
        help="OCR preprocessing: none, grayscale, threshold, adaptive, denoise",
    ),
):
    """Run the pipeline on one or more .mp4 files and save individual reports."""
    videos: list[Path] = []
    for arg in videos_or_dir:
        if arg.is_dir():
            videos.extend(sorted(arg.glob("*.mp4")))
        elif arg.exists():
            videos.append(arg)
        else:
            logger.error(f"File not found: {arg}")
            raise typer.Exit(1)

    if not videos:
        logger.error("No .mp4 files found")
        raise typer.Exit(1)

    models = _resolve_models(model)
    if not models:
        logger.error("No valid VLM models specified")
        raise typer.Exit(1)

    model_names = ", ".join(c.name for c in models)
    logger.info(f"VLM models: [{model_names}]  strategy: {strategy}")

    total = len(videos)
    for i, v in enumerate(videos, 1):
        logger.info(f"\n[{i}/{total}] Processing {v.name} ...")
        try:
            _run_single(v, jump_sec, models, strategy, gpu, batch_size, preprocessing)
            logger.info(f"[{i}/{total}] \u2705 {v.name} completed")
        except Exception as e:
            logger.error(f"[{i}/{total}] \u274c {v.name} failed: {e}")

    logger.info(f"\nBatch completed \u2014 {total} video(s) processed")


if __name__ == "__main__":
    app()
