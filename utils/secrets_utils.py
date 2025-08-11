import os
import streamlit as st


def get_secrets(mode):
    if mode == 'local':
        secrets = {
            'CONNECTION_STRING': os.getenv("CONNECTION_STRING"),
            'TTS_KEY': os.getenv("TTS_KEY"),
            'GEMINI_KEY': os.getenv("GEMINI_KEY")
        }
    elif mode == 'streamlit':
        secrets = {
            'CONNECTION_STRING': st.secrets["CONNECTION_STRING"],
            'TTS_KEY': st.secrets["TTS_KEY"],
            'GEMINI_KEY': st.secrets["GEMINI_KEY"]
        }
    else:
        secrets = {
            'CONNECTION_STRING': os.environ.get("CONNECTION_STRING"),
            'TTS_KEY': os.environ.get("TTS_KEY"),
            'GEMINI_KEY': os.environ.get("GEMINI_KEY")
        }
    secrets['TTS_REGION'] = "westeurope"
    return secrets
