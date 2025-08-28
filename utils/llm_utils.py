from google import genai
from google.genai import types
from utils.prompts import get_prompt
import json
import re
import ast


class LLM:
    def __init__(self, session):
        self.key = session["secrets"]["GEMINI_KEY"]
        self.client = genai.Client(api_key=self.key)
        self.model_name = session["model_name"]
        self.language = session["language"]
        self.prompt = get_prompt(self.language)

    def response(self, contents, generation_config):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generation_config
        )
        return response.text.strip()

    def get_full_response(self, image_data, audio_base64):
        full_prompt = f"""
                    {self.prompt}
                    Response to user question (audio) based on the image provided.
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
        response = self.response(contents, generation_config)
        output = self._parse_response(response)
        return output

    def _parse_response(self, raw_response):
        output = {"raw_response": raw_response}
        response_text = ""
        object_list = []
        is_list = False

        clean_response = raw_response.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:].strip().removesuffix('```')

        try:
            data = json.loads(clean_response)
            response_text = data.get("response", "")
            objects = data.get("search_objects", [])
            if isinstance(objects, list):
                object_list = [str(item) for item in objects]
                is_list = True
            else:
                output["warning"] = f"""
                    Error: JSON parsed correctly, but returned objects are not a list.
                    Raw response: {raw_response}
                """

        except json.JSONDecodeError:
            response_match = re.search(r'"response":\s*"(.*?)"', raw_response, re.DOTALL)
            if response_match:
                response_text = response_match.group(1).replace("\\n", "\n").replace("\\t", "\t")
            else:
                try:
                    # remove parts that look like JSON structure
                    response_text = re.sub(r'["{,:]\s*".*?"\s*[:\]}]', '', raw_response, flags=re.DOTALL)
                except:
                    pass
            objects_match = re.search(r'"search_objects":\s*(\[.*?])', raw_response, re.DOTALL)
            if objects_match:
                is_list, object_list = self._parse_list(objects_match.group(1))
            output["warning"] = f"Error decoding JSON. Raw response: {raw_response}"
        output["response_text"] = response_text
        output["object_list"] = object_list
        output["is_list"] = is_list
        return output

    def _parse_list(self, s):
        try:
            s_eval = ast.literal_eval(s)
            is_list = isinstance(s_eval, list)
            list_processed = [str(s) for s in s_eval] if is_list else []
            return is_list, list_processed
        except:
            return False, []
