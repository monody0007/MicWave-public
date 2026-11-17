"""Optional LLM post-processors (lazy-loaded SDKs so core imports stay light)."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLMProcessor(ABC):
    @abstractmethod
    async def process_text(
        self, text: str, prompt: str, model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    def process_text_sync(
        self, text: str, prompt: str, model: Optional[str] = None
    ) -> str:
        pass


class GeminiProcessor(LLMProcessor):
    def __init__(self, default_model: str = "gemini-1.5-pro"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self.default_model = default_model

    async def process_text(
        self, text: str, prompt: str, model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info("Using model: %s for processing", model_name)
        genai_model = self._genai.GenerativeModel(model_name)
        response = await genai_model.generate_content_async(all_prompt, stream=True)
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    def process_text_sync(
        self, text: str, prompt: str, model: Optional[str] = None
    ) -> str:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info("Using model: %s for sync processing", model_name)
        genai_model = self._genai.GenerativeModel(model_name)
        response = genai_model.generate_content(all_prompt)
        return response.text


class GPTProcessor(LLMProcessor):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        from openai import AsyncOpenAI, OpenAI

        self.async_client = AsyncOpenAI(api_key=api_key)
        self.sync_client = OpenAI(api_key=api_key)
        self.default_model = "gpt-4"

    async def process_text(
        self, text: str, prompt: str, model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info("Using model: %s for processing", model_name)
        response = await self.async_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": all_prompt}],
            stream=True,
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def process_text_sync(
        self, text: str, prompt: str, model: Optional[str] = None
    ) -> str:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info("Using model: %s for sync processing", model_name)
        response = self.sync_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": all_prompt}],
        )
        return response.choices[0].message.content


def get_llm_processor(model: str) -> LLMProcessor:
    normalized = model.lower()
    if normalized.startswith(("gemini", "gemini-")):
        return GeminiProcessor(default_model=normalized)
    if normalized.startswith(("gpt-", "o1-")):
        return GPTProcessor()
    raise ValueError(f"Unsupported model type: {normalized}")
