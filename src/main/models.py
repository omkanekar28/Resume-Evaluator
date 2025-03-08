import torch
from llama_cpp import Llama


class GGUFModel:
    """
    Class to handle the GGUF model.
    """

    def __init__(self, gguf_model_path: str, system_prompt: str, context_window_size: int, 
                 verbose: bool = False) -> None:
        """
        Initializes the model and its relevant parameters.
        """
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.system_prompt = system_prompt
            self.model = Llama(
                model_path=gguf_model_path,
                n_gpu_layers= -1 if self.device=='cuda' else 0,
                n_ctx=context_window_size,
                verbose=verbose
            )
            if self.device == 'cuda':
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
