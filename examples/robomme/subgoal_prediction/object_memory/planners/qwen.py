from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError

os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "64")
os.environ.setdefault("FPS_MAX_FRAMES", "10")

DEFAULT_MODEL_DIR = Path("runs/ckpts/vlm_subgoal_predictor/qwenvl/Qwen3-VL-4B-Instruct")
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass
class InferenceResult:
    parsed: BaseModel
    attempts: list[dict[str, Any]]


def extract_json(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if text[match.start() + end :].strip():
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No standalone JSON object found")


def _response_text(response: Any) -> str:
    choice = response[0].choices[0] if isinstance(response, list) else response.choices[0]
    content = choice.message.content
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected Swift response content: {type(content).__name__}")
    return content


class QwenBackend:
    """One reusable local Qwen3-VL engine for both split planner calls."""

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        *,
        max_retries: int = 2,
        seed: int = 7,
    ) -> None:
        from swift.llm import PtEngine
        import torch

        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise RuntimeError(f"Qwen checkpoint not found: {self.model_dir}")
        self.max_retries = max_retries
        self.seed = seed
        print(f"[object-memory] Loading Qwen3-VL from {self.model_dir}")
        self.engine = PtEngine(
            model_id_or_path=str(self.model_dir),
            adapters=None,
            attn_impl="sdpa",
            torch_dtype=torch.float16,
        )

    def infer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        schema: type[SchemaT],
        max_tokens: int,
    ) -> InferenceResult:
        from swift.llm import InferRequest
        from swift.llm import RequestConfig

        attempts: list[dict[str, Any]] = []
        correction = ""
        for _ in range(self.max_retries):
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        user_prompt
                        + "\n\nJSON SCHEMA\n"
                        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
                        + correction
                    ),
                },
            ]
            response = self.engine.infer(
                [InferRequest(messages=messages, images=image_paths)],
                request_config=RequestConfig(
                    max_tokens=max_tokens,
                    temperature=0,
                    seed=self.seed,
                ),
            )
            raw = _response_text(response)
            attempt: dict[str, Any] = {"messages": messages, "raw_output": raw}
            attempts.append(attempt)
            try:
                parsed = schema.model_validate(extract_json(raw))
                attempt["parsed_output"] = parsed.model_dump(mode="json")
                return InferenceResult(parsed=parsed, attempts=attempts)
            except (ValidationError, ValueError) as error:
                attempt["validation_error"] = str(error)
                correction = (
                    "\n\nCORRECTION REQUIRED\n"
                    "Your previous response was invalid. Return a corrected JSON object only.\n"
                    f"Validation error: {error}\nPrevious response: {raw}"
                )
        raise RuntimeError(f"Qwen returned no valid {schema.__name__}: {attempts[-1]['validation_error']}")
