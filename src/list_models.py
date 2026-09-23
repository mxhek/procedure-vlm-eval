from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = genai.Client()

for m in client.models.list():
    if "generateContent" in (m.supported_actions or []) and "flash" in m.name:
        print(m.name)