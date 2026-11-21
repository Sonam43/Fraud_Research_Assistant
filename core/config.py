"""Read configuration safely from environment variables or Streamlit secrets."""

import os
import streamlit as st


def get_setting(name: str, default: str = "") -> str:
    """Environment variables override Streamlit secrets, if both are present."""
    env_value = os.environ.get(name)
    if env_value:
        return env_value
    try:
        value = st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        value = default
    except Exception as exc:
        # Some Streamlit versions raise StreamlitSecretNotFoundError when
        # no secrets.toml file exists. Do not hide unrelated errors.
        if type(exc).__name__ == "StreamlitSecretNotFoundError":
            value = default
        else:
            raise
    return str(value) if value is not None else default


def openrouter_is_configured() -> bool:
    return bool(get_setting("OPENROUTER_API_KEY").strip())
