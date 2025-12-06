import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Safety: fail early if key is missing
# if not OPENAI_API_KEY:
#     raise ValueError("Missing OPENAI_API_KEY in .env file")

# Default model
MODEL_NAME = "gpt-5"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
