import base64
import json
import time
from pathlib import Path

import httpx
from loguru import logger

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "moondream"  # suportados: moondream, llava, minicpm-v, bakllava

PROMPT_TEMPLATE = """You are an exam proctoring assistant. Analyze this screenshot from a student's screen during an exam.

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


def _encode_image(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OllamaVLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE,
        timeout: int = 120,
        max_retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Analyse a single suspect frame
    # ------------------------------------------------------------------
    def analyse(self, image_path: str | Path) -> dict:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        b64 = _encode_image(path)
        payload = {
            "model": self.model,
            "prompt": PROMPT_TEMPLATE,
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
                last_err = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Ollama call failed after {1 + self.max_retries} attempts: {last_err}"
        )

    # ------------------------------------------------------------------
    # Parse the model response into structured JSON
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Batch analyse all suspects in a directory
    # ------------------------------------------------------------------
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

        logger.info(f"VLM analysis: {len(paths)} suspect(s) with model '{self.model}' ...")
        results = []
        for p in paths:
            logger.info(f"  Analysing {p.name} ...")
            verdict = self.analyse(p)
            verdict["file"] = str(p)
            results.append(verdict)
            icon = "⚠️  INFRACTION" if verdict["infraction"] else "✅  CLEAR"
            logger.info(f"    -> {icon}  |  {verdict['reason']}")

        return results

    # ------------------------------------------------------------------
    # Save verdicts to JSON
    # ------------------------------------------------------------------
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
