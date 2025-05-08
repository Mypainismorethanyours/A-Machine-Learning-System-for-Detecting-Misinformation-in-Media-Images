# Backend for handling model inferencing on user uploaded image
# This component of the ML system is defined as a FastAPI app endpoint

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

#=========================== App Startup Process =============================#
#FastAPI app is initialized -> loadmodel() function call -> model components loaded into memory -> model set to eval mode 

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
    version="1.0.0"
)

# Function to load the model and its components
def load_model():
    """Load the model and initialize + return all necessary components."""
    try:
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./output/Qwen2.5-VL-3B-Instruct/checkpoint-600")
        
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and processor
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Load base model from HuggingFace 
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
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

# to check that the model has been loaded and is ready 
@app.get("/health")
def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return {
        "status": "healthy" if model_components is not None else "degraded",
        "model_loaded": model_components is not None,
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