"""Groq LLM client for the construction risk analysis system."""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqClient:
    """Wrapper around the Groq API client."""

    def __init__(self, model: str = "openai/gpt-oss-120b") -> None:
        """Initialize the Groq client."""
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Please set it in the .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate a response from the Groq model."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content or ""