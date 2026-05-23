import os
import torch
import requests
import json
from typing import Optional
from llama_cpp import Llama
from google import genai
from google.genai import types


class GGUFModel:
    """
    Class to handle the GGUF model.
    """

    def __init__(self, gguf_model_path: str, system_prompt: str, context_window_size: int, 
                 verbose: bool = False, n_batch: int = 4, device = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """
        Initializes the model and its relevant parameters.
        """
        try:
            self.system_prompt = system_prompt
            self.model = Llama(
                model_path=gguf_model_path,
                n_gpu_layers= -1 if device=='cuda' else 0,
                n_batch=n_batch,
                n_ctx=context_window_size,
                verbose=verbose
            )
            if device == 'cuda':
                print(f"Model located at {gguf_model_path} loaded successfully on GPU.")
            else:
                print(f"Model located at {gguf_model_path} loaded successfully on CPU.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occured while trying to load the GGUF model: {str(e)}")
    
    def perform_inference(self, instruction_prompt: str) -> str:
        """
        Performs inference on the given instruction prompt and returns the model output.
        """
        try:
            messages = [
                {"role": "user", "content": f"{instruction_prompt}"}
            ]
            if self.system_prompt is not None:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            output = self.model.create_chat_completion(
                messages=messages,
            )
            text = output['choices'][0]['message']['content']
            return text
        except Exception as e:
            raise RuntimeError(f"An unexpected error occured while trying to perform inference: {str(e)}")


class GoogleGenaiModel:
    """
    Class to handle Google Generative AI models (Gemini) via the google-genai SDK.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str, 
        api_key: Optional[str] = None,
        include_thoughts: Optional[bool] = False,
    ) -> None:
        """
        Initializes the Google Generative AI model and its relevant parameters.
        """
        try:
            self.model_name = model_name
            self.system_prompt = system_prompt

            resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "No API key provided. Pass api_key= or set the GOOGLE_API_KEY environment variable."
                )

            self.client = genai.Client(api_key=resolved_key)

            # Build reusable GenerateContentConfig
            self._config = types.GenerateContentConfig(
                system_instruction=self.system_prompt if self.system_prompt else None, 
                thinking_config=types.ThinkingConfig(include_thoughts=include_thoughts) if include_thoughts else None
            )
            print(f"Google GenAI model '{self.model_name}' initialised successfully.")

        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while trying to initialise the Google GenAI model: {str(e)}"
            )

    def perform_inference(self, instruction_prompt: str) -> str:
        """
        Performs inference on the given instruction prompt and returns the model output.

        Args:
            instruction_prompt: The user-facing prompt to send to the model.

        Returns:
            The model's text response as a string.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=instruction_prompt,
                config=self._config,
            )
            return response.text

        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while trying to perform inference: {str(e)}"
            )


class OpenrouterModel:
    """
    Class to handle Openrouter models.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str, 
        api_key: Optional[str] = None,
        openrouter_api_timeout: int = 300,
    ) -> None:
        """
        Initializes the Openrouter model and its relevant parameters.
        """
        try:
            self.model_name = model_name
            self.system_prompt = system_prompt

            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "No API key provided. Pass api_key= or set the OPENROUTER_API_KEY environment variable."
                )
            self.openrouter_api_timeout = openrouter_api_timeout
            print(f"Openrouter model '{self.model_name}' initialised successfully.")

        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while trying to initialise the Openrouter model: {str(e)}"
            )

    def perform_inference(self, instruction_prompt: str) -> str:
        """
        Performs inference on the given instruction prompt and returns the model output.

        Args:
            instruction_prompt: The user-facing prompt to send to the model.

        Returns:
            The model's text response as a string.
        """
        try:
            messages = [
                {"role": "system", "content": f"{self.system_prompt}"},
                {"role": "user", "content": f"{instruction_prompt}"}
            ]
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "",
                    "X-Title": ""
                },
                data=json.dumps({
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 4096,
                }), 
                timeout=self.openrouter_api_timeout
            )
            response = response.json()
            merged_response = ""

            if "error" in response:
                raise RuntimeError(f"Openrouter API returned an error: {response['error']}")

            for output_dict in response['choices']:
                reasoning_text = output_dict.get("message", {}).get("reasoning")
                if reasoning_text:
                    merged_response += f"<think>\n{reasoning_text}\n</think>\n"
                merged_response += f"{output_dict['message']['content']}"
            
            return merged_response.strip()

        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while trying to perform inference: {str(e)}"
            )