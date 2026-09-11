from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel

from ..schemas import DecisionOutput
from ..schemas import PerceptionOutput
from ..schemas import SubgoalDecision

DEFAULT_CACHE_DIR = Path("/home/ed1116/.cache/huggingface")
DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"
MODEL_SLUG = "gemma-4-26b-a4b-it"


@dataclass
class InferenceResult:
    parsed: BaseModel
    attempts: list[dict[str, Any]]


def _construct_response(
    schema: type[BaseModel],
    value: dict[str, Any],
) -> BaseModel:
    if issubclass(schema, PerceptionOutput):
        return schema.model_validate(value)
    if schema is DecisionOutput:
        return DecisionOutput.model_construct(
            subgoal_completed=value["subgoal_completed"],
            completion_evidence=value["completion_evidence"],
            next_subgoal=SubgoalDecision.model_construct(**value["next_subgoal"]),
        )
    raise TypeError(f"Unsupported unchecked output schema: {schema.__name__}")


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


def _resolve_model_path(model_id_or_path: str | Path) -> Path:
    path = Path(model_id_or_path).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    try:
        snapshot = snapshot_download(
            repo_id=str(model_id_or_path),
            cache_dir=DEFAULT_CACHE_DIR,
            local_files_only=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"Gemma checkpoint is not available in {DEFAULT_CACHE_DIR}: {model_id_or_path}"
        ) from error
    return Path(snapshot).resolve()


def _user_content(user_prompt: str, image_paths: list[str]) -> list[dict[str, str]]:
    if not image_paths:
        return [{"type": "text", "text": user_prompt}]
    parts = user_prompt.split("<image>")
    if len(parts) != len(image_paths) + 1:
        return [
            *({"type": "image", "url": path} for path in image_paths),
            {"type": "text", "text": user_prompt},
        ]
    content: list[dict[str, str]] = []
    for part, path in zip(parts, image_paths, strict=False):
        if part:
            content.append({"type": "text", "text": part})
        content.append({"type": "image", "url": path})
    if parts[-1]:
        content.append({"type": "text", "text": parts[-1]})
    return content


class GemmaBackend:
    """One reusable local Gemma 4 engine for both split planner calls."""

    def __init__(
        self,
        model_id_or_path: Path | str = DEFAULT_MODEL_ID,
        *,
        seed: int = 7,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForMultimodalLM  # noqa: PLC0415
        from transformers import AutoProcessor  # noqa: PLC0415

        self.model_path = _resolve_model_path(model_id_or_path)
        self.seed = seed
        torch.manual_seed(seed)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[object-memory] Loading Gemma 4 from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()

    def infer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        schema: type[BaseModel],
        max_tokens: int,
    ) -> InferenceResult:
        import torch  # noqa: PLC0415

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": _user_content(user_prompt, image_paths),
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        raw = self.processor.decode(
            output[0][input_length:],
            skip_special_tokens=True,
        )
        value = extract_json(raw)
        parsed = _construct_response(schema, value)
        attempt = {
            "messages": messages,
            "raw_output": raw,
            "parsed_output": value,
        }
        return InferenceResult(parsed=parsed, attempts=[attempt])
