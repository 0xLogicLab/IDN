"""Optional local Transformers provider for IDN runtime packages."""

from __future__ import annotations

import gc
import json
from collections import OrderedDict
from threading import RLock
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from idn.framework.types import InferenceRequest, ModelPackage
from idn.models.loader import ChatModel


class ProviderError(RuntimeError):
    """Raised when a model provider cannot serve a request."""


class ModelProviderRegistry:
    """Dispatch model packages to different inference backends."""

    def __init__(self, providers: Optional[Mapping[str, Any]] = None) -> None:
        self._providers: Dict[str, Any] = dict(providers or {})

    def register(self, provider_id: str, provider: Any) -> None:
        if not provider_id.strip():
            raise ValueError("provider_id cannot be empty")
        if provider_id in self._providers:
            raise ValueError("provider already registered: {0}".format(provider_id))
        if not callable(getattr(provider, "generate", None)):
            raise TypeError("provider must implement generate(package, request)")
        self._providers[provider_id] = provider

    def generate(self, package: ModelPackage, request: InferenceRequest) -> str:
        provider = self._providers.get(package.provider_id)
        if provider is None:
            raise ProviderError(
                "no provider registered for package {0}: {1}".format(
                    package.package_id, package.provider_id
                )
            )
        return provider.generate(package, request)


class TransformersProvider:
    """Lazy local-model provider backed by IDN's existing ``ChatModel``.

    ``max_loaded_models=1`` is a conservative starting point for limited-memory
    devices. This synchronous prototype serializes model generation.
    """

    def __init__(
        self,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        local_files_only: bool = True,
        max_loaded_models: int = 1,
    ) -> None:
        if max_loaded_models <= 0:
            raise ValueError("max_loaded_models must be positive")
        self.dtype = dtype
        self.device = device
        self.local_files_only = local_files_only
        self.max_loaded_models = int(max_loaded_models)
        self._models: OrderedDict[str, ChatModel] = OrderedDict()
        self._lock = RLock()

    def generate(self, package: ModelPackage, request: InferenceRequest) -> str:
        with self._lock:
            model = self._get_model(package)
            return model.generate(
                messages=request.as_dict_messages(),
                max_new_tokens=request.max_new_tokens,
            )

    def _get_model(self, package: ModelPackage) -> ChatModel:
        if not package.model_path:
            raise ProviderError(
                "TransformersProvider needs a local model_path for {0}".format(
                    package.package_id
                )
            )
        model = self._models.get(package.package_id)
        if model is not None:
            self._models.move_to_end(package.package_id)
            return model

        while len(self._models) >= self.max_loaded_models:
            _, evicted = self._models.popitem(last=False)
            evicted.model = None
            evicted.tokenizer = None
            evicted.processor = None
            del evicted
            gc.collect()
            _empty_cuda_cache()

        model = ChatModel(
            model_path=package.model_path,
            adapter_path=package.adapter_path,
            dtype=self.dtype,
            device=self.device,
            local_files_only=self.local_files_only,
        ).load()
        self._models[package.package_id] = model
        return model


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class OpenAICompatibleProvider:
    """Call a vLLM/SGLang or other OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 300.0,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self._opener = opener or urlopen

    def generate(self, package: ModelPackage, request: InferenceRequest) -> str:
        model_id = package.backend_model_id or package.package_id
        payload = {
            "model": model_id,
            "messages": request.as_dict_messages(),
            "max_tokens": request.max_new_tokens,
            "temperature": 0,
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        endpoint = self.base_url
        if not endpoint.endswith("/v1"):
            endpoint += "/v1"
        endpoint += "/chat/completions"
        http_request = Request(endpoint, data=body, headers=headers, method="POST")

        try:
            response = self._opener(http_request, timeout=self.timeout_seconds)
            with response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ProviderError(
                "inference endpoint returned HTTP {0}: {1}".format(exc.code, detail)
            ) from exc
        except URLError as exc:
            raise ProviderError("could not reach inference endpoint: {0}".format(exc)) from exc
        except OSError as exc:
            raise ProviderError("inference endpoint I/O failed: {0}".format(exc)) from exc
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("invalid chat-completion response: {0}".format(exc)) from exc

        try:
            text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("chat-completion response has no assistant text") from exc
        if not isinstance(text, str):
            raise ProviderError("chat-completion content is not plain text")
        return text
