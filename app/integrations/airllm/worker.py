"""Chat_Software AirLLM worker: an OpenAI-compatible chat endpoint over AirLLM's layer streaming.

Runs ONLY under the isolated AirLLM Python (never the app venv); Chat_Software starts and stops it:

    <airllm-python> worker.py --config <home>/catalog.json --port <port>

One model is resident at a time and inference is serialized. Models come only from the local
catalog written by Chat_Software (downloaded or registered checkpoints); the Hugging Face hub is
kept offline here. Every response reports the engine and backend so results are never mistaken for
another runtime. ``/cancel`` stops the current generation at the next token.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("airllm-worker")
CHECKPOINT_FILES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                    "pytorch_model.bin.index.json")


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=60_000)


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # OpenAI SDKs may send fields this worker does not use
    model: str = Field(min_length=1, max_length=200)
    messages: list[Message] = Field(min_length=1, max_length=80)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0, le=2)
    stream: Literal[False] = False


class Engine:
    def __init__(self, config: dict):
        self.config = config
        self.models = {m["id"]: m for m in config.get("models", [])}
        self.backend = config.get("backend", "cpu")
        if self.backend not in {"cpu", "metal", "cuda"}:
            raise ValueError("backend must be cpu, metal or cuda")
        os.environ["AIRLLM_BACKEND"] = self.backend
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.loaded_id: Optional[str] = None
        self.loaded = None
        self.progress: dict = {"phase": "idle"}
        self.last_error: Optional[str] = None

    def load(self, model_id: str):
        if model_id not in self.models:
            raise HTTPException(404, "Model is not in the local AirLLM catalog")
        if self.loaded_id == model_id:
            return self.loaded
        path = Path(self.models[model_id]["path"])
        if not path.is_dir() or not any((path / name).exists() for name in CHECKPOINT_FILES):
            raise HTTPException(503, "Model checkpoint is missing or incomplete on disk")
        executor = getattr(self.loaded, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        self.loaded, self.loaded_id = None, None
        gc.collect()
        from airllm import AutoModel

        kwargs = {"hf_token": False, "delete_original": False,
                  "max_seq_len": int(self.models[model_id].get("context_tokens", 4096))}
        if self.backend == "metal":
            config = json.loads((path / "config.json").read_text())
            arch = (config.get("architectures") or [""])[0]
            if config.get("tie_word_embeddings"):
                raise HTTPException(422, "AirLLM's Metal (MLX) backend cannot run models with tied input/output "
                                         "embeddings (for example SmolLM2). Switch AIRLLM_BACKEND to cpu.")
            if arch not in {"LlamaForCausalLM", "MistralForCausalLM"}:
                raise HTTPException(422, f"AirLLM's Metal (MLX) backend only runs Llama-family models, not {arch}. "
                                         "Switch AIRLLM_BACKEND to cpu.")
            # The MLX path stores .npz layer shards; keep them apart from the CPU/CUDA safetensors shards
            # (both default to <model>/splitted_model, and each backend cannot read the other's format).
            kwargs["layer_shards_saving_path"] = str(path / "airllm-mlx")
        if self.backend in {"cpu", "cuda"}:
            import torch

            if self.backend == "cpu":
                torch.set_num_threads(int(self.config.get("cpu_threads", 4)))
            kwargs.update(device="cpu" if self.backend == "cpu" else "cuda:0",
                          dtype=torch.float32 if self.backend == "cpu" else torch.float16,
                          prefetching=bool(self.config.get("prefetching", True)))
        self.progress = {"phase": "loading", "model": model_id}
        self.loaded = AutoModel.from_pretrained(str(path), **kwargs)
        self.loaded_id = model_id
        return self.loaded

    def complete(self, body: CompletionRequest) -> dict:
        with self.lock:
            started = time.monotonic()
            self.cancel_event.clear()
            model = self.load(body.model)
            tokenizer = model.tokenizer
            messages = [m.model_dump() for m in body.messages]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:  # base models without a chat template
                prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\n\nassistant:"
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            limit = min(body.max_tokens or self.config.get("max_output_tokens", 512),
                        int(self.config.get("max_output_tokens", 512)))
            context = int(self.models[body.model].get("context_tokens", 4096))
            if len(ids) + limit > context:
                raise HTTPException(422, f"Prompt plus output exceeds this model's {context}-token context; "
                                         "start a new conversation or shorten the request.")
            deadline = float(self.config.get("generation_timeout_seconds", 1800))
            self.progress = {"phase": "generating", "model": body.model, "prompt_tokens": len(ids)}
            if self.backend == "metal":
                import mlx.core as mx

                tokens: list[int] = []
                for token in model.model_generate(mx.array([ids]), temperature=body.temperature):
                    value = int(token.item())
                    if self.cancel_event.is_set():
                        raise HTTPException(409, "Generation cancelled")
                    if time.monotonic() - started > deadline:
                        raise HTTPException(504, "Generation reached its time limit")
                    if value == tokenizer.eos_token_id:
                        break
                    tokens.append(value)
                    self.progress["generated_tokens"] = len(tokens)
                    if len(tokens) >= limit:
                        break
            else:
                import torch
                from transformers import StoppingCriteria, StoppingCriteriaList

                engine = self
                prompt_length = len(ids)

                class Stop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **extra):
                        engine.progress["generated_tokens"] = int(input_ids.shape[1] - prompt_length)
                        return engine.cancel_event.is_set() or time.monotonic() - started > deadline

                kwargs = {"max_new_tokens": limit, "do_sample": body.temperature > 0,
                          "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                          "eos_token_id": tokenizer.eos_token_id, "use_cache": True,
                          "stopping_criteria": StoppingCriteriaList([Stop()])}
                if body.temperature > 0:
                    kwargs["temperature"] = body.temperature
                tensor = torch.tensor([ids], device=model.running_device)
                with torch.inference_mode():
                    output = model.generate(tensor, attention_mask=torch.ones_like(tensor), **kwargs)
                if self.cancel_event.is_set():
                    raise HTTPException(409, "Generation cancelled")
                if time.monotonic() - started > deadline:
                    raise HTTPException(504, "Generation reached its time limit")
                tokens = output[0, len(ids):].tolist()
                if tokens and tokens[-1] == tokenizer.eos_token_id:
                    tokens = tokens[:-1]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            elapsed = round(time.monotonic() - started, 3)
            self.progress = {"phase": "idle", "last_elapsed_seconds": elapsed, "last_tokens": len(tokens)}
            return {
                "id": "airllm-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                "model": body.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "length" if len(tokens) >= limit else "stop"}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(tokens),
                          "total_tokens": len(ids) + len(tokens)},
                "engine": {"name": "airllm", "backend": self.backend, "elapsed_seconds": elapsed},
            }


def create_app(config: dict, engine: Optional[Engine] = None) -> FastAPI:
    service = engine or Engine(config)
    app = FastAPI(title="Chat_Software AirLLM worker", version="0.1.0")
    app.state.engine = service

    @app.get("/health")
    def health():
        return {"status": "ready", "engine": "airllm", "backend": service.backend,
                "loaded_model": service.loaded_id, "busy": service.lock.locked(),
                "models": list(service.models), "progress": service.progress, "last_error": service.last_error}

    @app.post("/cancel")
    def cancel():
        service.cancel_event.set()
        return {"status": "cancellation_requested"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "local-airllm"}
                                           for m in service.models]}

    @app.post("/v1/chat/completions")
    def complete(body: CompletionRequest):
        try:
            return service.complete(body)
        except HTTPException as exc:
            service.progress = {"phase": "idle"}
            service.last_error = str(exc.detail)
            raise
        except Exception as exc:
            logger.exception("AirLLM inference failed")
            service.progress = {"phase": "idle"}
            service.last_error = f"{exc.__class__.__name__}: {str(exc)[:300]}"
            raise HTTPException(503, f"AirLLM failed: {service.last_error}") from None

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    uvicorn.run(create_app(config), host="127.0.0.1", port=args.port, timeout_graceful_shutdown=3,
                log_level="warning")


if __name__ == "__main__":
    main()
