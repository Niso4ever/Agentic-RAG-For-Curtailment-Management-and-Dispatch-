from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env from current folder
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

print("Loaded API key:", api_key[:8] + "..." if api_key else "NONE FOUND")

client = OpenAI(api_key=api_key)

resp = client.responses.create(
    model="gpt-5",
    input="test GPT5"
)

print(resp)
