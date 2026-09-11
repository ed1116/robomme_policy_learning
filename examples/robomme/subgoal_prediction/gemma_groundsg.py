from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess
from typing import ClassVar

from subgoal_prediction.gemini.api import BaseModel

DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"
RPC_PREFIX = "__GEMMA_GROUNDSG_RPC__"


class GemmaGroundSGModel(BaseModel):
    """Local Gemma backend with the same stateful interface as Gemini GroundSG."""

    _shared_worker: ClassVar[subprocess.Popen[str] | None] = None
    _atexit_registered: ClassVar[bool] = False

    def __init__(
        self,
        *,
        save_dir: str,
        task_id: str,
        model_name: str = DEFAULT_MODEL_ID,
        task_goal: str,
        subgoal_type: str = "grounded_subgoal",
        image_size: tuple = (256, 256),
        max_new_tokens: int = 512,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.worker: subprocess.Popen[str] | None = None
        super().__init__(
            save_dir=save_dir,
            task_id=task_id,
            model_name=model_name,
            task_goal=task_goal,
            subgoal_type=subgoal_type,
            image_size=image_size,
        )

    def init_model(self, system_prompt: str, model_name: str) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        worker_python = Path(
            os.environ.get(
                "GEMMA_GROUNDSG_WORKER_PYTHON",
                repo_root / ".venv-gemma4" / "bin" / "python",
            )
        )
        worker_script = Path(__file__).with_name("gemma_groundsg_worker.py")
        if not worker_python.is_file():
            raise FileNotFoundError(f"Gemma worker Python not found: {worker_python}")

        if self._shared_worker is None or self._shared_worker.poll() is not None:
            self._start_shared_worker(worker_python, worker_script)
        self.worker = self._shared_worker
        self._rpc(
            {
                "command": "init",
                "model_id_or_path": model_name or DEFAULT_MODEL_ID,
                "system_prompt": system_prompt,
                "max_new_tokens": self.max_new_tokens,
            }
        )

    @classmethod
    def _start_shared_worker(
        cls,
        worker_python: Path,
        worker_script: Path,
    ) -> None:
        cls._shared_worker = subprocess.Popen(
            [str(worker_python), "-u", str(worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if not cls._atexit_registered:
            atexit.register(cls.shutdown_shared_worker)
            cls._atexit_registered = True

    @classmethod
    def shutdown_shared_worker(cls) -> None:
        worker = cls._shared_worker
        cls._shared_worker = None
        if worker is None or worker.poll() is not None:
            return
        if worker.stdin is not None:
            worker.stdin.close()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.terminate()
            worker.wait(timeout=10)

    def _rpc(self, request: dict) -> dict:
        if self.worker is None or self.worker.stdin is None or self.worker.stdout is None:
            raise RuntimeError("Gemma worker is not running")
        if self.worker.poll() is not None:
            raise RuntimeError(f"Gemma worker exited with code {self.worker.returncode}")

        self.worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.worker.stdin.flush()
        while True:
            line = self.worker.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"Gemma worker exited before replying (code {self.worker.poll()})"
                )
            if not line.startswith(RPC_PREFIX):
                print(line, end="")
                continue
            response = json.loads(line[len(RPC_PREFIX) :])
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "Gemma worker request failed"))
            return response

    def _process_media(
        self,
        *,
        media_type: str,
        paths: list[str],
        text_query: str,
    ) -> str:
        response = self._rpc(
            {
                "command": "infer",
                "media_type": media_type,
                "paths": paths,
                "text_query": text_query,
            }
        )
        raw_output = response["raw_output"]
        self.conversation_history.append(
            {
                "turn": len(self.conversation_history) + 1,
                "type": media_type,
                "path": [os.path.basename(path) for path in paths]
                if len(paths) > 1
                else os.path.basename(paths[0]),
                "query": text_query,
                "response": raw_output,
            }
        )
        return raw_output

    def _process_image(
        self,
        image_path: str | list[str],
        text_query: str = "What should the robot do in this situation?",
    ) -> str:
        paths = image_path if isinstance(image_path, list) else [image_path]
        return self._process_media(
            media_type="image",
            paths=paths,
            text_query=text_query,
        )

    def _process_video(
        self,
        video_path: str,
        text_query: str = "What should the robot do based on this video?",
    ) -> str:
        return self._process_media(
            media_type="video",
            paths=[video_path],
            text_query=text_query,
        )

    def _process_text(self, user_query: str) -> str:
        response = self._rpc(
            {
                "command": "infer",
                "media_type": "text",
                "paths": [],
                "text_query": user_query,
            }
        )
        raw_output = response["raw_output"]
        self.conversation_history.append(
            {
                "turn": len(self.conversation_history) + 1,
                "type": "text",
                "query": user_query,
                "response": raw_output,
            }
        )
        return raw_output

    def clear_uploaded_files(self) -> None:
        # The local worker has no uploaded remote files. Keep the weights resident;
        # the next episode's init request replaces its system prompt and chat history.
        self.worker = None
