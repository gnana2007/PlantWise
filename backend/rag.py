
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


def generate_answer(query, context):
    try:
        prompt = f"""
You are PlantIQ, an AI-powered plant knowledge assistant.

User Question:
{query}

Retrieved Plant Information:
{context}

Instructions:
- Answer using only the provided information.
- Mention relevant plant names.
- Explain why they are suitable.
- Keep the answer concise and factual.
- Use bullet points when appropriate.
- If the information is insufficient, say so.

Answer:
"""

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"Error generating response: {str(e)}"
