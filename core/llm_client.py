"""OpenRouter adapter. Live evidence-grounded QA will be added in a later step."""

from openai import OpenAI
from core.config import get_setting

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_client() -> OpenAI:
    key = get_setting("OPENROUTER_API_KEY").strip()
    if not key:
        raise ValueError("OPENROUTER_API_KEY is missing. Add it to Streamlit secrets.")
    return OpenAI(
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        timeout=45.0,
        max_retries=1,
    )


def test_connection() -> tuple[str, str]:
    """One real, low-token request. Only called when the user clicks Test."""
    model = get_setting("OPENROUTER_MODEL", "openrouter/free").strip()
    if not model:
        raise ValueError("OPENROUTER_MODEL is missing.")
    result = create_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply exactly with: API OK"}],
        max_tokens=30,
    )
    content = result.choices[0].message.content or "(Empty response)"
    return model, content
