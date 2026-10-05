"""Shared Gemini chat model setup."""

from config import GEMINI_API_KEY, GEMINI_MODEL


def gemini_chat_model(temperature: float = 0, model: str | None = None):
    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY in .env to enable Gemini")
    from langchain_openai import ChatOpenAI
    target_model = model or "gemini-3.1-flash-lite"
    candidate_models = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]
    fallback_models = [m for m in candidate_models if m != target_model]

    primary = ChatOpenAI(
        model=target_model,
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=temperature,
        max_retries=2,
    )
    fallbacks = [
        ChatOpenAI(
            model=m,
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=temperature,
            max_retries=2,
        )
        for m in fallback_models
    ]
    return primary.with_fallbacks(fallbacks)
