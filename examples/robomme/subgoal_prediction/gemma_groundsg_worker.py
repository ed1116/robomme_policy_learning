from __future__ import annotations

import json
from pathlib import Path
import sys

DEFAULT_CACHE_DIR = Path("/home/ed1116/.cache/huggingface")
RPC_PREFIX = "__GEMMA_GROUNDSG_RPC__"


def _resolve_model_path(model_id_or_path: str) -> Path:
    path = Path(model_id_or_path).expanduser()
    if path.is_dir():
        return path.resolve()

    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=model_id_or_path,
            cache_dir=DEFAULT_CACHE_DIR,
            local_files_only=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"Gemma checkpoint is not available in {DEFAULT_CACHE_DIR}: "
            f"{model_id_or_path}"
        ) from error
    return Path(snapshot).resolve()


class GemmaGroundSGWorker:
    def __init__(
        self,
        *,
        model_id_or_path: str,
        system_prompt: str,
        max_new_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM
        from transformers import AutoProcessor

        torch.manual_seed(7)
        model_path = _resolve_model_path(model_id_or_path)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[gemma-groundsg] Loading Gemma 4 from {model_path}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.reset(
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )

    def reset(self, *, system_prompt: str, max_new_tokens: int) -> None:
        self.max_new_tokens = max_new_tokens
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        ]

    def infer(
        self,
        *,
        media_type: str,
        paths: list[str],
        text_query: str,
    ) -> str:
        import torch

        content = [{"type": "text", "text": text_query}]
        if media_type in {"image", "video"}:
            content.extend(
                {"type": media_type, "path": path}
                for path in paths
            )
        self.messages.append({"role": "user", "content": content})

        inputs = self.processor.apply_chat_template(
            self.messages,
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
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        raw_output = self.processor.decode(
            output[0][input_length:],
            skip_special_tokens=True,
        ).strip()
        self.messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": raw_output}],
            }
        )
        return raw_output


def _reply(response: dict) -> None:
    print(RPC_PREFIX + json.dumps(response, ensure_ascii=False), flush=True)


def main() -> None:
    worker = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request["command"]
            if command == "init":
                if worker is None:
                    worker = GemmaGroundSGWorker(
                        model_id_or_path=request["model_id_or_path"],
                        system_prompt=request["system_prompt"],
                        max_new_tokens=request["max_new_tokens"],
                    )
                else:
                    worker.reset(
                        system_prompt=request["system_prompt"],
                        max_new_tokens=request["max_new_tokens"],
                    )
                _reply({"ok": True})
            elif command == "infer":
                if worker is None:
                    raise RuntimeError("Worker has not been initialized")
                raw_output = worker.infer(
                    media_type=request["media_type"],
                    paths=request["paths"],
                    text_query=request["text_query"],
                )
                _reply({"ok": True, "raw_output": raw_output})
            elif command == "close":
                _reply({"ok": True})
                return
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as error:
            _reply({"ok": False, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
