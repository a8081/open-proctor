from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx
from loguru import logger

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "moondream"

DEFAULT_PROMPT = """You are an exam proctoring assistant. Analyze this screenshot from a student's screen during an exam.

You MUST respond with ONLY valid JSON, no other text:
{{
  "infraction": true,
  "reason": "Brief explanation of what you see"
}}

Set infraction to TRUE only if you see clear evidence of:
- Messaging apps (Discord, WhatsApp, Telegram, Google Chat)
- AI tools (ChatGPT, GitHub Copilot, Gemini, Claude)
- Unauthorized browser tabs with prohibited content

Set infraction to FALSE if the screen shows:
- The exam content itself (questions, text editor, PDF)
- Nothing clearly unauthorized

Be conservative: if unsure, set infraction to false."""


@dataclass
class VLMConfig:
    name: str
    model: str
    prompt: str = DEFAULT_PROMPT


DEFAULT_MODELS = [
    VLMConfig(name="Moondream", model="moondream"),
    VLMConfig(name="LLaVA", model="llava"),
    VLMConfig(name="MiniCPM-V", model="minicpm-v"),
    VLMConfig(name="BakLLaVA", model="bakllava"),
]


class AggregationStrategy(str, Enum):
    SINGLE = "single"
    CONSENSUS = "consensus"
    MAJORITY = "majority"
    ANY = "any"


def _encode_image(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OllamaVLM:
    def __init__(
        self,
        model: str | list[VLMConfig] = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE,
        timeout: int = 120,
        max_retries: int = 2,
        strategy: AggregationStrategy | str = AggregationStrategy.SINGLE,
        active_model: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.strategy = AggregationStrategy(strategy)

        if isinstance(model, str):
            self.models = [VLMConfig(name=model, model=model)]
            self.active_model = model
        else:
            self.models = model
            self.active_model = active_model or (model[0].name if model else "moondream")

    def _resolve_config(self, model_name: str | None = None) -> VLMConfig:
        name = model_name or self.active_model
        for c in self.models:
            if c.name == name or c.model == name:
                return c
        return self.models[0]

    def _call_model(self, image_path: Path, config: VLMConfig) -> dict:
        b64 = _encode_image(image_path)
        payload = {
            "model": config.model,
            "prompt": config.prompt,
            "images": [b64],
            "stream": False,
        }

        last_err = None
        for attempt in range(1 + self.max_retries):
            try:
                resp = httpx.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                raw = resp.json()["response"].strip()
                return self._parse(raw)

            except Exception as e:
                err_str = str(e).lower()
                # Fail fast if model doesn't support images
                if "does not support image input" in err_str:
                    raise RuntimeError(
                        f"[{config.name}] Model '{config.model}' does not support image input. "
                        "Select a vision-capable model (e.g. moondream, llava, minicpm-v, bakllava)."
                    )
                last_err = e
                logger.warning(f"[{config.name}] Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2**attempt)

        raise RuntimeError(
            f"[{config.name}] Ollama call failed after {1 + self.max_retries} attempts: {last_err}"
        )

    @staticmethod
    def _parse(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Model did not return valid JSON:\n{raw[:200]}")
            return {"infraction": False, "reason": "Could not parse model response"}

        infraction = bool(data.get("infraction", False))
        reason = str(data.get("reason", ""))
        return {"infraction": infraction, "reason": reason}

    def _aggregate(self, results: list[dict]) -> dict:
        infraction_count = sum(1 for r in results if r["infraction"])
        total = len(results)

        if self.strategy == AggregationStrategy.CONSENSUS:
            infraction = infraction_count == total
        elif self.strategy == AggregationStrategy.MAJORITY:
            infraction = infraction_count > total / 2
        elif self.strategy == AggregationStrategy.ANY:
            infraction = infraction_count > 0
        else:
            infraction = results[0]["infraction"] if results else False

        primary_idx = 0
        reasons = [r["reason"] for r in results]
        if infraction:
            for i, r in enumerate(results):
                if r["infraction"]:
                    primary_idx = i
                    break

        return {
            "infraction": infraction,
            "reason": results[primary_idx]["reason"],
            "per_model": results,
            "strategy": self.strategy.value,
            "infraction_count": infraction_count,
            "total_models": total,
        }

    def analyse(self, image_path: str | Path) -> dict:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        if self.strategy == AggregationStrategy.SINGLE:
            config = self._resolve_config()
            result = self._call_model(path, config)
            result["file"] = str(path)
            result["model"] = config.name
            return result

        # Multi-model analysis
        all_results = []
        for cfg in self.models:
            r = self._call_model(path, cfg)
            r["model"] = cfg.name
            all_results.append(r)

        aggregated = self._aggregate(all_results)
        aggregated["file"] = str(path)
        return aggregated

    def analyse_directory(
        self,
        suspects_dir: str | Path = "data/suspects",
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> list[dict]:
        suspects_dir = Path(suspects_dir)
        paths = sorted(
            p for p in suspects_dir.iterdir() if p.suffix.lower() in image_extensions
        )
        if not paths:
            logger.warning(f"No suspect images found in {suspects_dir}")
            return []

        model_names = ", ".join(c.name for c in self.models)
        logger.info(
            f"VLM analysis: {len(paths)} suspect(s) "
            f"models=[{model_names}] strategy={self.strategy.value} ..."
        )

        results = []
        for p in paths:
            logger.info(f"  Analysing {p.name} ...")
            verdict = self.analyse(p)
            results.append(verdict)
            icon = "⚠️ INFRACTION" if verdict["infraction"] else "✅ CLEAR"
            logger.info(f"    -> {icon}  |  {verdict['reason']}")

        return results

    @staticmethod
    def save_verdicts(
        verdicts: list[dict],
        output_path: str | Path = "data/suspects/verdicts.json",
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(verdicts, f, indent=2)
        logger.info(f"Verdicts saved to {output_path}")
