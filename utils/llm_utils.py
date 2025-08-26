from google import genai
from google.genai import types
from utils.prompts import PROMPT_ASSIST, PROMPT_SEARCH
import ast


class LLM:
    def __init__(self, session):
        self.key = session["secrets"]["GEMINI_KEY"]
        self.client = genai.Client(api_key=self.key)
        self.model_name = session["model_name"]
        self.language = session["language"]

    def response(self, contents, generation_config):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generation_config
        )
        return response.text.strip()

    def get_image_audio_description(self, image_data, audio_base64):
        full_prompt = f"""
            {PROMPT_ASSIST}
            Response to user question (audio) based on the image provided.
            **Language of response:** {self.language}
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
        return self.response(contents, generation_config)

    def _parse_list(self, s):
        try:
            s_eval = ast.literal_eval(s)
            is_list = isinstance(s_eval, list)
            s_eval = [str(s) for s in s_eval] if is_list else []
            return is_list, s_eval
        except:
            return False, []

    def search_audio(self, audio_base64):
        full_prompt = PROMPT_SEARCH
        contents = [
            types.Content(
                parts=[
                    types.Part(text=full_prompt),
                    types.Part(inline_data=types.Blob(mime_type="audio/wav", data=audio_base64))
                ]
            )
        ]
        generation_config = types.GenerateContentConfig(
            temperature=0,
            # thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        resp = self.response(contents, generation_config)
        is_list, objects = self._parse_list(resp)
        return is_list, objects, resp
