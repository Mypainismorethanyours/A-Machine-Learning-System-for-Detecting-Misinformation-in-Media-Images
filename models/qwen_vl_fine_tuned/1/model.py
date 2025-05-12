#for every model version theres a directory associated with the model version
import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import json
import os
import logging

#according to the lab 

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model."""
        # This method loads the model and moves it to the device spcfd in args
        #then puts the model in inference mode 
        #this fucntion is runs when triton starts and the model from the directory thats passed is loaded  
        self.model_config = model_config = json.loads(args['model_config'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model configuration
        self.base_model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./output/Qwen2.5-VL-3B-Instruct/checkpoint-600")
        
        # Load tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, 
            use_fast=False, 
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(self.base_model_id)
        
        # LoRA config
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=True,
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )
        
        # Load model with optimizations
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16
        }
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_id,
            **model_kwargs
        )
        
        if os.path.exists(self.checkpoint_path):
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.checkpoint_path,
                config=self.lora_config
            )
        
        self.model.eval()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Model initialized successfully")

    def execute(self, requests):
        """Execute the model for inference requests."""
        responses = []
        
        for request in requests:
            # Get input tensors
            image = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            text = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            
            # Convert inputs to appropriate format
            image_np = image.as_numpy()
            text_str = text.as_numpy()[0].decode('utf-8')
            
            # Process image
            image_pil = Image.fromarray(image_np)
            
            # Create message structure
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": text_str}
                ]
            }]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            # Prepare model inputs
            inputs = self.processor(
                text=[text], 
                images=image_inputs, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
            # Decode output
            prediction = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[-1]:], 
                skip_special_tokens=True
            )[0]
            
            # Extract label and reasoning
            label, reasoning = self._extract_label_and_reasoning(prediction)
            
            # Create output tensors
            prediction_tensor = pb_utils.Tensor(
                "prediction", 
                np.array([prediction.encode('utf-8')], dtype=np.object_)
            )
            label_tensor = pb_utils.Tensor(
                "label", 
                np.array([label.encode('utf-8')], dtype=np.object_)
            )
            reasoning_tensor = pb_utils.Tensor(
                "reasoning", 
                np.array([reasoning.encode('utf-8')], dtype=np.object_)
            )
            
            # Create response
            response = pb_utils.InferenceResponse(
                output_tensors=[prediction_tensor, label_tensor, reasoning_tensor]
            )
            responses.append(response)
        
        return responses

    def _extract_label_and_reasoning(self, text: str) -> tuple:
        """Extract label and reasoning from model output."""
        text_lower = text.lower()
        
        # Determine the label ( the reason for why its detect as synthesized)
        if any(phrase in text_lower for phrase in [
            "has been manipulated", "is manipulated", 
            "synthetic", "generated", "fake", 
            "altered", "doctored"
        ]):
            label = "fake"
        elif any(phrase in text_lower for phrase in [
            "has not been manipulated", "authentic", 
            "original", "genuine", "real", "unaltered"
        ]):
            label = "real"
        else:
            label = "unknown"
        
        # Extract the models reasoning for tje prediciton result 
        reasoning = text
        if ". " in text:
            parts = text.split(". ", 1)
            if len(parts) > 1:
                reasoning = parts[1].strip()
        
        return label, reasoning

    def finalize(self):
        """Clean up when the model is being unloaded."""
        self.logger.info("Cleaning up model resources")
        del self.model
        torch.cuda.empty_cache() 