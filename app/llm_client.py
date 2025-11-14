"""Utility for creating an OpenAI Responses client with graceful fallbacks."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # Module missing when not installed in the current interpreter
    OpenAI = None  # type: ignore

# Load environment variables if a .env file exists
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Default to a generally available model; allow override via MODEL_NAME env.
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client: Optional["OpenAI"] = None

if OPENAI_API_KEY and OpenAI:
    client = OpenAI(api_key=OPENAI_API_KEY)
