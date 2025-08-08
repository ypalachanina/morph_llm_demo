from google import genai
from google.genai import types
from utils.prompts import PROMPT_ASSIST
import os
import streamlit as st

# GEMINI_KEY = os.environ.get("GEMINI_KEY")
# GEMINI_KEY = os.getenv("GEMINI_KEY")
GEMINI_KEY = st.secrets["GEMINI_KEY"]
client = genai.Client(api_key=GEMINI_KEY)


def get_audio_description(image_data, audio_base64, model_name, language='English'):
    full_prompt = f"""
        {PROMPT_ASSIST}
        Response to user question (audio) based on the image provided.
        **Language of response:** {language}
    """
    contents = [
        types.Content(
            parts=[
                types.Part(text=full_prompt),
                types.Part(inline_data=types.Blob(mime_type="image/png", data=image_data)),
                types.Part(inline_data=types.Blob(mime_type="audio/wav", data=audio_base64))
            ]
        )
    ]
    generation_config = types.GenerateContentConfig(
        temperature=0,
        # thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generation_config
    )
    return response.text.strip()
