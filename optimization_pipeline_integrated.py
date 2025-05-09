import torch
import time
import json
import os
import mlflow
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, write_to_textfile
import numpy as np
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info
from PIL import Image
import psutil
import GPUtil

# Prometheus metrics
registry = CollectorRegistry()

optimization_duration = Histogram(
    'model_optimization_duration_seconds',
    'Time spent evaluating optimization',
    ['model_type', 'optimization_type'],
    registry=registry
)

model_inference_time = Gauge(
    'model_inference_time_ms',
    'Model inference time in milliseconds',
    ['model_type', 'optimization_type', 'model_version'],
    registry=registry
)

model_memory_usage_bytes = Gauge(
    'model_memory_usage_bytes',
    'Model memory usage in bytes',
    ['model_type', 'optimization_type', 'model_version'],
    registry=registry
)

@dataclass
class OptimizationResult:
    name: str
    model_type: str
    model_version: str
    optimization_type: str
    model_size_mb: float
    inference_time_ms: float
    memory_usage_mb: float
    accuracy_metrics: Dict[str, float]
    config: Dict[str, Any]
    load_time_s: float
    throughput_images_per_sec: float
    timestamp: str
    gpu_utilization_percent: float
    cuda_memory_allocated_mb: float
    mlflow_run_id: Optional[str] = None

class QwenModelOptimizer:
    def __init__(self, base_model_id: str, checkpoint_path: str, test_dataset_path: str, 
                 model_version: str, mlflow_experiment_name: str = None):
        self.base_model_id = base_model_id
        self.checkpoint_path = checkpoint_path
        self.test_dataset_path = test_dataset_path
        self.model_version = model_version
        self.mlflow_experiment_name = mlflow_experiment_name or "Model Optimization"
        self.results = []
        
        # Setup directories
        self.results_dir = f"./optimization_results/{model_version}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load tokenizer and processor once
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        
        # LoRA config for loading fine-tuned model
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=True,
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )
    
    def load_model_with_optimization(self, optimization_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load model with specified optimization"""
        config = config or {}
        
        # Model loading arguments
        model_kwargs = {
            "device_map": "auto",
        }
        
        # Apply optimization-specific configurations
        if optimization_type == "baseline":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif optimization_type == "quantization_8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_quant_type="nf4",
                bnb_8bit_use_double_quant=True
            )
        elif optimization_type == "quantization_4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif optimization_type == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif optimization_type == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif optimization_type == "flash_attention":
            model_kwargs["torch_dtype"] = torch.bfloat16
            # Flash attention will be enabled after model loading
        
        # Load base model
        load_start = time.time()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_id,
            **model_kwargs
        )
        
        # Enable flash attention if requested
        if optimization_type == "flash_attention" and hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = True
        
        # Load LoRA weights if checkpoint exists
        if os.path.exists(self.checkpoint_path):
            model = PeftModel.from_pretrained(model, self.checkpoint_path, config=self.lora_config)
        
        model.eval()
        load_time = time.time() - load_start
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return {
            "model": model,
            "processor": self.processor,
            "tokenizer": self.tokenizer,
            "device": device,
            "load_time": load_time,
            "optimization_type": optimization_type
        }
    
    def measure_performance(self, model_components: Dict, num_samples: int = 10) -> Dict[str, float]:
        """Measure model performance metrics"""
        model = model_components["model"]
        processor = model_components["processor"]
        device = model_components["device"]
        
        # Load test dataset
        test_samples = self._load_test_samples(num_samples)
        
        inference_times = []
        memory_measurements = []
        gpu_utilizations = []
        
        # Warm up
        if test_samples:
            self._run_inference(test_samples[0], model, processor, device)
        
        for sample in test_samples:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Measure GPU utilization
            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_utilizations.append(gpus[0].load * 100)
            
            # Measure inference time
            start_time = time.time()
            self._run_inference(sample, model, processor, device)
            end_time = time.time()
            
            inference_times.append((end_time - start_time) * 1000)  # ms
            
            # Measure memory
            if torch.cuda.is_available():
                memory_measurements.append(torch.cuda.max_memory_allocated(device) / (1024 * 1024))  # MB
            else:
                process = psutil.Process()
                memory_measurements.append(process.memory_info().rss / (1024 * 1024))  # MB
        
        # Calculate model size
        model_size_mb = self._get_model_size(model)
        
        # Calculate metrics
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        throughput = 1000 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            "model_size_mb": model_size_mb,
            "inference_time_ms": avg_inference_time,
            "memory_usage_mb": np.mean(memory_measurements) if memory_measurements else 0,
            "throughput_images_per_sec": throughput,
            "gpu_utilization_percent": np.mean(gpu_utilizations) if gpu_utilizations else 0,
            "cuda_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
        }
    
    def _run_inference(self, sample: Dict, model, processor, device):
        """Run inference on a single sample"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": "Has this image been manipulated or synthesized?"}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        return generated_ids
    
    def _load_test_samples(self, num_samples: int) -> List[Dict]:
        """Load test samples from dataset"""
        samples = []
        
        # Try to load from test dataset
        if os.path.exists(self.test_dataset_path):
            with open(self.test_dataset_path, 'r') as f:
                test_data = json.load(f)
            
            for item in test_data[:num_samples]:
                input_content = item["conversations"][0]["value"]
                image_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
                
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                    samples.append({"image": image, "item": item})
        
        # If no samples loaded, create dummy samples
        if not samples:
            for i in range(num_samples):
                dummy_image = Image.new('RGB', (448, 448), color='white')
                samples.append({"image": dummy_image, "item": {}})
        
        return samples
    
    def _get_model_size(self, model) -> float:
        """Calculate model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def evaluate_optimization(self, optimization_type: str, config: Dict[str, Any] = None) -> OptimizationResult:
        """Evaluate a single optimization configuration"""
        config = config or {}
        
        # Start MLflow run
        mlflow.set_experiment(self.mlflow_experiment_name)
        with mlflow.start_run(nested=True, run_name=f"optimization_{optimization_type}_{self.model_version}") as run:
            # Log parameters
            mlflow.log_param("optimization_type", optimization_type)
            mlflow.log_param("model_version", self.model_version)
            mlflow.log_params(config)
            
            # Load model with optimization
            with optimization_duration.labels(
                model_type="qwen2_vl",
                optimization_type=optimization_type
            ).time():
                model_components = self.load_model_with_optimization(optimization_type, config)
                
                # Measure performance
                metrics = self.measure_performance(model_components)
            
            # Log metrics to MLflow
            mlflow.log_metric("model_size_mb", metrics["model_size_mb"])
            mlflow.log_metric("inference_time_ms", metrics["inference_time_ms"])
            mlflow.log_metric("memory_usage_mb", metrics["memory_usage_mb"])
            mlflow.log_metric("throughput_images_per_sec", metrics["throughput_images_per_sec"])
            mlflow.log_metric("gpu_utilization_percent", metrics["gpu_utilization_percent"])
            mlflow.log_metric("load_time_s", model_components["load_time"])
            
            # Create result
            result = OptimizationResult(
                name=f"{optimization_type}_{json.dumps(config)}",
                model_type="qwen2_vl",
                model_version=self.model_version,
                optimization_type=optimization_type,
                model_size_mb=metrics["model_size_mb"],
                inference_time_ms=metrics["inference_time_ms"],
                memory_usage_mb=metrics["memory_usage_mb"],
                accuracy_metrics={"accuracy": 0.0},  # You can add actual accuracy evaluation here
                config=config,
                load_time_s=model_components["load_time"],
                throughput_images_per_sec=metrics["throughput_images_per_sec"],
                timestamp=datetime.now().isoformat(),
                gpu_utilization_percent=metrics["gpu_utilization_percent"],
                cuda_memory_allocated_mb=metrics["cuda_memory_allocated_mb"],
                mlflow_run_id=run.info.run_id
            )
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(result)
            
            self.results.append(result)
            return result
    
    def _update_prometheus_metrics(self, result: OptimizationResult):
        """Update Prometheus metrics"""
        labels = {
            'model_type': result.model_type,
            'optimization_type': result.optimization_type,
            'model_version': result.model_version
        }
        
        model_inference_time.labels(**labels).set(result.inference_time_ms)
        model_memory_usage_bytes.labels(**labels).set(result.memory_usage_mb * 1024 * 1024)
    
    def run_optimization_suite(self, optimizations_to_test: List[Dict[str, Any]]) -> List[OptimizationResult]:
        """Run multiple optimization configurations"""
        results = []
        
        for opt_config in optimizations_to_test:
            opt_type = opt_config["type"]
            config = opt_config.get("config", {})
            
            print(f"Evaluating optimization: {opt_type} with config: {config}")
            result = self.evaluate_optimization(opt_type, config)
            results.append(result)
            
            # Save intermediate results
            self.save_results()
        
        return results
    
    def save_results(self):
        """Save all results"""
        # Save detailed results
        results_file = os.path.join(self.results_dir, 'optimization_results.json')
        results_data = [asdict(r) for r in self.results]
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save Prometheus metrics
        metrics_file = os.path.join(self.results_dir, 'metrics.prom')
        write_to_textfile(metrics_file, registry)
        
        # Generate report
        report = self.generate_report()
        report_file = os.path.join(self.results_dir, 'optimization_report.md')
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save deployment recommendation
        self._save_deployment_recommendation()
    
    def generate_report(self) -> str:
        """Generate markdown report"""
        report = f"# Model Optimization Report\n\n"
        report += f"Model Version: {self.model_version}\n"
        report += f"Base Model: {self.base_model_id}\n"
        report += f"Checkpoint: {self.checkpoint_path}\n"
        report += f"Timestamp: {datetime.now().isoformat()}\n\n"
        
        # Summary table
        report += "## Optimization Summary\n\n"
        report += "| Optimization | Model Size (MB) | Inference (ms) | Memory (MB) | Throughput (img/s) | GPU Util (%) | Load Time (s) |\n"
        report += "|-------------|----------------|---------------|-------------|-------------------|--------------|---------------|\n"
        
        for result in self.results:
            report += f"| {result.optimization_type} | {result.model_size_mb:.2f} | {result.inference_time_ms:.2f} | "
            report += f"{result.memory_usage_mb:.2f} | {result.throughput_images_per_sec:.2f} | "
            report += f"{result.gpu_utilization_percent:.1f} | {result.load_time_s:.2f} |\n"
        
        # Best configurations
        if self.results:
            best_speed = min(self.results, key=lambda r: r.inference_time_ms)
            best_memory = min(self.results, key=lambda r: r.memory_usage_mb)
            best_throughput = max(self.results, key=lambda r: r.throughput_images_per_sec)
            
            report += "\n## Best Configurations\n\n"
            report += f"- **Fastest Inference**: {best_speed.optimization_type} ({best_speed.inference_time_ms:.2f} ms)\n"
            report += f"- **Lowest Memory**: {best_memory.optimization_type} ({best_memory.memory_usage_mb:.2f} MB)\n"
            report += f"- **Highest Throughput**: {best_throughput.optimization_type} ({best_throughput.throughput_images_per_sec:.2f} img/s)\n"
        
        return report
    
    def _save_deployment_recommendation(self):
        """Save deployment recommendation"""
        if not self.results:
            return
        
        # Score optimizations (you can adjust weights)
        def score_optimization(result):
            # Lower is better
            return (result.inference_time_ms * 0.4 + 
                   result.memory_usage_mb * 0.3 + 
                   (1000 / result.throughput_images_per_sec) * 0.3)
        
        best_overall = min(self.results, key=score_optimization)
        
        recommendation = {
            "model_version": self.model_version,
            "recommended_optimization": best_overall.optimization_type,
            "recommended_config": best_overall.config,
            "expected_metrics": {
                "inference_time_ms": best_overall.inference_time_ms,
                "memory_usage_mb": best_overall.memory_usage_mb,
                "throughput_images_per_sec": best_overall.throughput_images_per_sec,
                "gpu_utilization_percent": best_overall.gpu_utilization_percent
            },
            "mlflow_run_id": best_overall.mlflow_run_id,
            "timestamp": datetime.now().isoformat()
        }
        
        rec_file = os.path.join(self.results_dir, 'deployment_recommendation.json')
        with open(rec_file, 'w') as f:
            json.dump(recommendation, f, indent=2)