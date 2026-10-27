import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

print("API KEY LOADED:", bool(os.getenv("GOOGLE_API_KEY")))

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

print("Before Request")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Hello, give me one short sentence."
)

print("After Request")
print(response.text)
