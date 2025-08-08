from io import BytesIO
import base64
import azure.cognitiveservices.speech as speechsdk
import os
import streamlit as st

# TTS_KEY = os.environ.get("TTS_KEY")
# TTS_KEY = os.getenv("TTS_KEY")
TTS_KEY = st.secrets["TTS_KEY"]
TTS_REGION = "westeurope"


def text_to_speech(text, language):
    voices = {
        "English": "en-GB-RyanNeural",
        "Nederlands": "nl-NL-FennaNeural",
        "Vlaams": "nl-BE-DenaNeural"
    }

    config = speechsdk.SpeechConfig(subscription=TTS_KEY, region=TTS_REGION)
    config.speech_synthesis_voice_name = voices[language]

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=config,
        audio_config=speechsdk.audio.AudioOutputConfig(filename=rf"C:\Users\Eugenia\Downloads\tts_{language}.wav")
    )

    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_bytes = BytesIO(result.audio_data)
        audio_base64 = base64.b64encode(result.audio_data).decode()
        return audio_bytes, audio_base64
    else:
        return None


def audio_to_base64(audio):
    buf = BytesIO()
    audio.export(buf, format="wav")
    wav_bytes = buf.getvalue()
    audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")
    return audio_base64
