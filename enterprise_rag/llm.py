"""Generation backends. All receive only evidence that already passed permission filtering."""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str, *, max_tokens: int = 700) -> str: ...


class OpenAICompatibleLLM:
    """Any OpenAI-style /chat/completions server: Ollama (http://localhost:11434/v1), llama.cpp server,
    vLLM, LM Studio. No OpenAI account is needed. Suggested CPU-friendly Qwen choices on an 8 GB laptop:
    qwen2.5:1.5b-instruct (fast) or qwen2.5:3b-instruct (better), via `ollama pull`."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 180.0):
        self.base_url, self.model, self.api_key, self.timeout = base_url.rstrip("/"), model, api_key, timeout
        self.name = f"openai-compatible:{model}"

    def generate(self, system: str, user: str, *, max_tokens: int = 700) -> str:
        import requests  # imported lazily so the core package has no hard dependency

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = requests.post(
            f"{self.base_url}/chat/completions", headers=headers, timeout=self.timeout,
            json={"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class LangChainLLM:
    """Adapter for any LangChain chat model (ChatOllama, ChatOpenAI pointed at a local server, ...)."""

    def __init__(self, chat_model, name: str = "langchain-chat-model"):
        self.chat_model, self.name = chat_model, name

    def generate(self, system: str, user: str, *, max_tokens: int = 700) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage  # optional dependency

        reply = self.chat_model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(getattr(reply, "content", reply)).strip()
