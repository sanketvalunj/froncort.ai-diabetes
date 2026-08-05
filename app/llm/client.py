# Client wrapper for interacting with the LLM API.
import json
import os

from config.settings import LLMSettings, settings as _app_settings


class LLMParseError(Exception):
    pass


class LLMClient:
    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self._client = None

    @property
    def client(self):
        if self._client is None:
            provider = self.settings.provider

            if provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Pull key from: app settings (loaded from .env) → raw environment
                api_key = _app_settings.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY is not set. Add it to .env locally "
                        "or to Render environment variables in production."
                    )
                self._client = ChatGoogleGenerativeAI(
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                    google_api_key=api_key,
                )

            elif provider == "xai":
                from langchain_xai import ChatXAI
                api_key = _app_settings.xai_api_key or os.environ.get("XAI_API_KEY", "")
                if not api_key:
                    raise ValueError(
                        "XAI_API_KEY is not set. Add it to .env locally "
                        "or to Render environment variables in production."
                    )
                self._client = ChatXAI(
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                    xai_api_key=api_key,
                )

            else:
                raise ValueError(
                    f"Unknown LLM provider: {provider!r}. "
                    f"Supported providers: google, xai"
                )

        return self._client

    def generate(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        response = self.client.invoke([HumanMessage(content=prompt)])
        return response.content

    def generate_structured(self, prompt: str) -> dict:
        raw = self.generate(prompt)
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            return json.loads(raw[start:end])
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMParseError(f"Failed to parse LLM response: {exc}") from exc
