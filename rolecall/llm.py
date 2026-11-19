"""Minimal client for any OpenAI-compatible /chat/completions endpoint with tool calling."""

import os
import time
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLM:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60
    calls: list[dict] = field(default_factory=list)  # per-call usage and latency, read by the evals

    @classmethod
    def from_env(cls) -> "LLM":
        missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.getenv(k)]
        if missing:
            raise SystemExit(f"set {', '.join(missing)} in .env (see .env.example)")
        return cls(os.environ["LLM_BASE_URL"].rstrip("/"), os.environ["LLM_API_KEY"], os.environ["LLM_MODEL"])

    def chat(self, messages: list[dict], tools: list[dict], retries: int = 3) -> dict:
        body = {"model": self.model, "messages": messages, "tools": tools, "temperature": 0}
        for attempt in range(retries):
            started = time.perf_counter()
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                resp.raise_for_status()
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
                continue
            data = resp.json()
            self.calls.append({"ms": (time.perf_counter() - started) * 1000, "usage": data.get("usage", {})})
            return data["choices"][0]["message"]
        raise RuntimeError("unreachable")
