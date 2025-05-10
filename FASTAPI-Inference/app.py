# Backend for handling model inferencing on user uploaded image
# This component of the ML system is defined as a FastAPI app endpoint
# 5/9/25 : Updated the file to support optimization strategies.
#The Updates process :
# Model is loaded with the optimization strategies that are recommended based on the results from the optimization strategies pipeline tests
#The recommended optimization strategies are loaded from a json file that gets updates with the recommended optimzations for the current model
#This ensures that the inference service uses the most optimal model / optimization configuration


# Import necessary dependencies
import torch
import os
import io
import json
import logging
import base64
from fastapi import FastAPI, HTTPException, File, UploadFile
from typing import Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from PIL import Image
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

#=========================== App Startup Process =============================#
#FastAPI app is initialized -> loadmodel() function call -> recomended optimizations loaded from json file loaded into memory -> model set to eval mode 
#




# Define metrics for production monitoring
REQUEST_COUNT = Counter(
    'model_prediction_requests_total', 
    'Total prediction requests',
    ['model_version', 'optimization_type', 'status']
)

REQUEST_LATENCY = Histogram(
    'model_prediction_duration_seconds', 
    'Prediction request latency',
    ['model_version', 'optimization_type']
)

MODEL_MEMORY_USAGE = Gauge(
    'model_memory_usage_mb',
    'Model memory usage in MB',
    ['model_version', 'optimization_type']
)

GPU_UTILIZATION = Gauge(
    'model_gpu_utilization_percent',
    'GPU utilization percentage',
    ['model_version', 'optimization_type']
)







# Model specific imports
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)

from peft import PeftModel, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request and response models
class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image data")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Full model reasoning and prediction text")
    label: str = Field(..., description="Extracted label (real, fake, or unknown)")
    reasoning: str = Field(..., description="Extracted reasoning from the model's response")
    processing_time: float = Field(..., description="Total processing time in seconds")

# Initialize the FastAPI app
app = FastAPI(
    title="ML System for Detecting-Misinformation-in-Media-Images",
    description="API endpoint for model inferencing on user submitted data",
    version="2.0.0" 
)

#load recommended optimization strategies based on the results from the optimization stratgies tests pipeline
class ModelConfig:
    ""Configuration for model optimiation""
    def _init_ (self):
        #Load con figuration from enviro/ file
        config_path = os.environ.get("MODEL_CONFIG_PATH", "./config.json")

        #Default configs
        self.default_config = {
            "optimization_type": "bf16",
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "checkpoint_path": "./output/Qwen2.5-VL-3B-Instruct/checkpoint-600",
            "optimization_config"
        }
        # Try to load configuration file
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                self.config = {**self.default_config, **file_config}
                logger.info(f"Loaded configuration from {config_path}")
        else:
            self.config = self.default_config
            logger.info("Using default configuration")
        
        # Override with environment variables
        self.optimization_type = os.environ.get("OPTIMIZATION_TYPE", self.config["optimization_type"])
        self.model_id = os.environ.get("MODEL_ID", self.config["model_id"])
        self.checkpoint_path = os.environ.get("CHECKPOINT_PATH", self.config["checkpoint_path"])

# Function to load the model + its components with optimization configuration
def load_model_with_optimization(config:ModelConfig):
    ""Load the model with the specified optim. strategies""
    try: 
        optimization_type = config.optimization_type
        model_id = config.model_id
        checkpoint_path = config.checkpoint_path

        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and processor
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id)


def load_model():
    """Load the model and initialize + return all necessary model components as a dict."""
    try:
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./output/Qwen2.5-VL-3B-Instruct/checkpoint-600")
        
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and processor
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id)

        #Model loading arguments bsed on optim type
        model_kwargs = {
            "device_map": "auto",
        }
        
        # Apply Optimization specific configs 
        if optimization_type == "baseline" or optimization_type == "bf16":
            model_kwargs["torch_dtype"] == torch.bfloat16
        elif optimization_type == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif optimization_type == "quantization_8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype = torch.float16,
                bnb_8bit_use_double_quant = True

            )
        elif optimization_type == "quantization_4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        elif optimization_type == "flash_attention":
            model_kwargs["torch_dtype"] = torch.bfloat16
            # Flash attention will be enabled after model loading
        
        else:
            logger.warning(f"Unknown optimization type: {optimization_type}, using bf16")
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        # Load base model from HuggingFace 
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )

        # Enable flash attention if requested
        if optimization_type == "flash_attention" and hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = True
            logger.info("Flash Attention 2 enabled")

        
        # Configure LoRA adapter for inference
        val_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=True,
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )
        
        # Load the fine-tuned model
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading fine-tuned model from {checkpoint_path}")
            peft_model = PeftModel.from_pretrained(model, checkpoint_path, config=val_config)
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}, using base model")
            peft_model = model
        
        # Set the model to evaluation mode
        peft_model.eval()

        # Log model configuration
        logger.info(f"Model loaded successfully with optimization: {optimization_type}")
        logger.info(f"Model dtype: {next(peft_model.parameters()).dtype}")
        logger.info(f"Device: {next(peft_model.parameters()).device}")
        
        # Return all the model components as a dictionary 
        return {
            "model": peft_model,
            "tokenizer": tokenizer,
            "processor": processor,
            "device": device
        }
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None
#Load configuration 
model_config = ModelConfig()


# Load model components into memory at startup (as opposed to when the request arrives)
logger.info("Loading model components...")
model_load_start = datetime.now()
model_components = load_model()
model_load_time = (datetime.now() - model_load_start).total_seconds()

# Error handling for unsuccessful loading of model components
if model_components is None:
    logger.error("Failed to load components. API will not function properly.")
else:
    logger.info(f"Model components loaded successfully in {model_load_time:.2f} seconds")

# Helper function to extract the model prediction: label and reasoning
def label_and_reasoning_extraction(text: str) -> Tuple[str, str]:
    """Extract image label and reasoning from the model prediction text"""
    
    full_text = text.strip()
    text_lower = full_text.lower()
    
    # Determine label based on keywords in the response
    if any(phrase in text_lower for phrase in ["has been manipulated", "is manipulated", 
                                               "synthetic", "generated", "fake", 
                                               "altered", "doctored"]):
        label = "fake"
    elif any(phrase in text_lower for phrase in ["has not been manipulated", "authentic", 
                                                 "original", "genuine", "real", "unaltered"]):
        label = "real"
    else:
        label = "unknown"
    
    # Extract reasoning - the full text is the reasoning
    reasoning = full_text
    
    # Try to extract just the reasoning part if we can identify a clear pattern
    if ". " in full_text:
        parts = full_text.split(". ", 1)
        if len(parts) > 1:
            reasoning = parts[1].strip()
    
    return label, reasoning

def run_inference(image):
    """Run model inference on image"""
    if model_components is None:
        raise ValueError("Model components not loaded")
    #create message structure with image and text
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Has this image been manipulated or synthesized?"}
        ]
    }]
    
    # Get model components
    processor = model_components["processor"]
    model = model_components["model"]
    device = model_components["device"]
    
    # Process text part of the message 
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #Extract the images from messages using the utility function ( in qwen_vl_utils.py)
    image_inputs, _ = process_vision_info(messages)
    #combine the image and text for model input 
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
    
    # Generate model output
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)
        # Get the result
        result = processor.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
    
    # Return the result
    return result

@app.post("/predict/file", response_model=PredictionResponse)
async def predict_image_file(file: UploadFile = File(...)):
    """Direct file upload endpoint"""
    if model_components is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Start timing
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Time file reading
        read_start = datetime.now()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        read_time = (datetime.now() - read_start).total_seconds()
        
        # Time inference
        inference_start = datetime.now()
        prediction_text = run_inference(image)
        inference_time = (datetime.now() - inference_start).total_seconds()
        
        # Time post-processing
        postprocess_start = datetime.now()
        label, reasoning = label_and_reasoning_extraction(prediction_text)
        postprocess_time = (datetime.now() - postprocess_start).total_seconds()
        
        # Calculate total processing time
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log timing breakdown
        logger.info(f"File upload timing - Read: {read_time:.3f}s, Inference: {inference_time:.3f}s, "
                   f"Post-process: {postprocess_time:.3f}s, Total: {total_processing_time:.3f}s")
        
        return PredictionResponse(
            prediction=prediction_text,
            label=label,
            reasoning=reasoning,
            processing_time=total_processing_time
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
def model_predict(request: ImageRequest):
    """Model Prediction Endpoint"""
    if model_components is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Start timing
    start_time = datetime.now()
    
    try:
        # Time image decoding
        decode_start = datetime.now()
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        decode_time = (datetime.now() - decode_start).total_seconds()
        
        # Time inference
        inference_start = datetime.now()
        prediction_text = run_inference(image)
        inference_time = (datetime.now() - inference_start).total_seconds()
        
        # Time post-processing
        postprocess_start = datetime.now()
        label, reasoning = label_and_reasoning_extraction(prediction_text)
        postprocess_time = (datetime.now() - postprocess_start).total_seconds()
        
        # Calculate total processing time
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log timing breakdown
        logger.info(f"Timing breakdown - Decode: {decode_time:.3f}s, Inference: {inference_time:.3f}s, "
                   f"Post-process: {postprocess_time:.3f}s, Total: {total_processing_time:.3f}s")
        
        return PredictionResponse(
            prediction=prediction_text,
            label=label,
            reasoning=reasoning,
            processing_time=total_processing_time
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Store startup time for health check
app_start_time = datetime.now()

# NEW: Endpoint to check the optimization statuss
@app.get("/optimization_info")
def get_optimization_info():
    """Get information about the current optimization configuration"""
    if model_components is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "optimization_type": model_components.get("optimization_type", "unknown"),
        "model_id": model_config.model_id,
        "checkpoint_path": model_config.checkpoint_path,
        "device": str(model_components.get("device", "unknown")),
        "model_dtype": str(next(model_components["model"].parameters()).dtype),
        "model_load_time": model_load_time
    }

#To check that the model has been loaded and is ready 
#This is updated to include optimization info 
@app.get("/health")
def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return {
        "status": "healthy" if model_components is not None else "degraded",
        "model_loaded": model_components is not None,
        "optimization_type": model_components.get("optimization_type","unkown")
        "model_load_time": model_load_time if model_components is not None else None,
        "uptime_seconds": uptime
    }

# basic api info 
@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "ML System for Detecting Misinformation in Media Images",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Submit image for manipulation detection (base64)",
            "/predict/file": "POST - Submit image file for manipulation detection",
            "/health": "GET - Check service health",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, debug=False)