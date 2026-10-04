"""Shared Gemini chat model setup."""

from config import GEMINI_API_KEY, GEMINI_MODEL


def gemini_chat_model(temperature: float = 0):
    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY in .env to enable Gemini")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
    )
