import json
from config.settings import LLMSettings


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

            if provider == "openai":
                from langchain_openai import ChatOpenAI
                self._client = ChatOpenAI(
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                )

            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                self._client = ChatAnthropic(
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                )

            elif provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._client = ChatGoogleGenerativeAI(
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                )

            elif provider == "xai":
                from langchain_xai import ChatXAI
                self._client = ChatXAI(
                    model=self.settings.model,
                    temperature=self.settings.temperature,
                )

            else:
                raise ValueError(
                    f"Unknown LLM provider: {provider!r}. "
                    f"Supported providers: openai, anthropic, google, xai"
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
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            return json.loads(raw[start:end])
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMParseError(f"Failed to parse LLM response: {exc}") from exc
