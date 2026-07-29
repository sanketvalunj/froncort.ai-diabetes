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
            if self.settings.provider == "openai":
                from langchain_openai import ChatOpenAI
                self._client = ChatOpenAI(model=self.settings.model,
                                          temperature=self.settings.temperature)
            elif self.settings.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                self._client = ChatAnthropic(model=self.settings.model,
                                             temperature=self.settings.temperature)
            else:
                raise ValueError(f"Unknown LLM provider: {self.settings.provider!r}")
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
