# defines a comprehensive model level optimization pipeline that is designed to 
#systematically test differnt optimization techniques on a train Qwen vision-language model
#based on the results the best optimization strategies are idetified and saved in a json file 
import torch
import time
import json
import os
import mlflow
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
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

#stores all of the metrics and resukts for each optimization test, we can then comparew the results in a standardized way
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
    latency_percentiles: Dict[str,float] = field(default_factory=dict)

#overview of proces that occurs 
# Loads the base model and checkpoint
# applies different optimizations
#measures performance, in relation to the optimization strategy applied that applied to the modle 
#generates a reoport based on the result of the optimization and the optimization with the best performncae relative to the requirmensts for our model are
#save + recommended as an optimiation strategy to apply 


class QwenModelOptimizer:
    def __init__(self, base_model_id: str, checkpoint_path: str, test_dataset_path: str, 
                 model_version: str, mlflow_experiment_name: str = None):
        self.base_model_id = base_model_id
        self.checkpoint_path = checkpoint_path
        self.test_dataset_path = test_dataset_path
        self.model_version = model_version
        self.mlflow_experiment_name = mlflow_experiment_name or "Model Optimization"
        self.results = []
        self.baseline_metrics = None
        
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

    # Before applying model level optimizations we want to establish a baseline meausuremnt of the model size, the accuracy (on test data), inference latency per a single sample, 
    # and the batch throughput per x number of samples
    #

    def baseline_measure_performance(self):

        """Establish a baseline performance measurement without any optimizations : float32, no quantization"""

        #load just the trained model itslef

        print("="*50)
        print("Measuring baseline inference performance")
        print("="*50)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using:{device}")

        #load base model fom hugg face

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_id,
            torch_dtype = torch.float32,
            device_map="auto"
        )

        # load weights for the fine tuned models
        if os.path.exists(self.checkpoint_path):
            model = PeftModel.from_pretrained(
                model, 
                self.checkpoint_path,  # "./output/Qwen2.5-VL-3B-Instruct/checkpoint-600"
                config=self.lora_config
            )
    
        model.eval()

        #measure the model size on the disk : doen via helper function 
        model_size = self._get_model_size(model)
        print(f"Model size: {model_size:.2f}MB")

        # Prepare test data
        #whats the path ?
        test_samples = self._load_test_samples(num_samples=1)
        single_sample = test_samples[0] if test_samples else None

        # Add accuracy measurement
        print("Measuring baseline accuracy...")
        accuracy_metrics = self.calculate_accuracy(
            {"model": model, "processor": self.processor, "device": device},
            num_samples=50  # Adjust based on your needs
        )
        print(f"Baseline Accuracy: {accuracy_metrics['accuracy']:.2f}%")
        
        # Measure inference latency on single sample ( we do this for 100 trials and then compute the agregate statistics)
        num_trials = 100
        latencies = []

        # Warm-up run
        with torch.no_grad():
            self._run_inference(single_sample, model, self.processor, device)

        # Actual measurement
        for i in range(num_trials):
            start_time = time.time()
            with torch.no_grad():
                _ = self._run_inference(single_sample, model, self.processor, device)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        

        # Calculate statistics (like in the lab)
        median_latency = np.median(latencies)
        percentile_95 = np.percentile(latencies, 95)
        percentile_99 = np.percentile(latencies, 99)
        
        
        print(f"Inference Latency (single sample, median): {median_latency:.1f} ms")
        print(f"Inference Latency (single sample, 95th percentile): {percentile_95:.1f} ms")
        print(f"Inference Latency (single sample, 99th percentile): {percentile_99:.1f} ms")
       
        # Add batch throughput measurement
        print("\nMeasuring batch throughput...")
        batch_metrics = self.measure_batch_throughput(
            {"model": model, "processor": self.processor, "device": device},
            batch_sizes=[1, 4, 8, 16]
        )

        print(f"Optimal batch size: {batch_metrics['optimal_batch_size']}")
        print(f"Maximum throughput: {batch_metrics['max_throughput_images_per_sec']:.2f} images/second")
        

        
        # Create baseline result
        baseline_result = OptimizationResult(
            name="baseline",
            model_type="qwen2_vl",
            model_version=self.model_version,
            optimization_type="baseline",
            model_size_mb=model_size,
            inference_time_ms=median_latency,
            memory_usage_mb=0,  # Will measure if needed
            accuracy_metrics=accuracy_metrics,
            config={
                "batch_throughput": batch_metrics['throughput_by_batch_size'],
                "optimal_batch_size": batch_metrics['optimal_batch_size']
            },
            load_time_s=0,
            throughput_images_per_sec=batch_metrics['max_throughput_images_per_sec'],
            timestamp=datetime.now().isoformat(),
            gpu_utilization_percent=0,
            cuda_memory_allocated_mb=0,
            latency_percentiles={
                "50th": median_latency,
                "95th": percentile_95,
                "99th": percentile_99
            }
            
        )
        
        self.baseline_metrics = baseline_result
        self.results.append(baseline_result)
        
        print("="*50)
        return baseline_result

    def calculate_accuracy(self, model_components: Dict, num_samples: int = 50) -> Dict[str, float]:
        """Calculate model accuracy on test samples"""
        model = model_components["model"]
        processor = model_components["processor"]
        device = model_components["device"]
        
        test_samples = self._load_test_samples(num_samples)
        
        correct = 0
        total = 0
        
        # Keywords for classification
        positive_keywords = ["manipulated", "synthesized", "edited", "fake", "generated", "artificial"]
        negative_keywords = ["real", "authentic", "original", "genuine", "unedited"]
        
        for sample in test_samples:
            _, generated_text, ground_truth = self._run_inference(
                sample, model, processor, device, return_text=True
            )
            
            if generated_text and ground_truth:
                generated_lower = generated_text.lower()
                
                # Simple keyword-based classification
                is_positive_pred = any(keyword in generated_lower for keyword in positive_keywords)
                is_negative_pred = any(keyword in generated_lower for keyword in negative_keywords)
                
                is_positive_truth = any(keyword in ground_truth for keyword in positive_keywords)
                
                if is_positive_pred and is_positive_truth:
                    correct += 1
                elif is_negative_pred and not is_positive_truth:
                    correct += 1
                
                total += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct
        }
    #helper function to measure the rate ate which the model can return predictions for batches of data 
    def measure_batch_throughput(self, model_components: Dict, batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[str, float]:
        """Measure actual batch throughput at different batch sizes"""
        model = model_components["model"]
        processor = model_components["processor"]
        device = model_components["device"]
        
        throughput_results = {}
        
        for batch_size in batch_sizes:
            print(f"Measuring throughput for batch size {batch_size}...")
            
            # Load enough samples for batching
            test_samples = self._load_test_samples(num_samples=batch_size * 5)  # 5 batches worth
            
            if len(test_samples) < batch_size:
                print(f"Not enough samples for batch size {batch_size}, skipping...")
                continue
            
            # Warm-up
            self._run_batch_inference(test_samples[:batch_size], model, processor, device)
            
            # Measure batch throughput
            num_batches = len(test_samples) // batch_size
            total_images = num_batches * batch_size
            
            start_time = time.time()
            
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                batch_samples = test_samples[batch_start:batch_end]
                
                self._run_batch_inference(batch_samples, model, processor, device)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate throughput (images per second)
            throughput = total_images / total_time
            throughput_results[f"batch_{batch_size}"] = throughput
            
            print(f"Batch size {batch_size}: {throughput:.2f} images/second")
        
        # Find optimal batch size
        optimal_batch_size = max(throughput_results.items(), key=lambda x: x[1])[0]
        max_throughput = max(throughput_results.values())
        
        return {
            "throughput_by_batch_size": throughput_results,
            "optimal_batch_size": optimal_batch_size,
            "max_throughput_images_per_sec": max_throughput
        }
    def _run_batch_inference(self, batch_samples: List[Dict], model, processor, device):
        """Run inference on a batch of samples"""
        batch_messages = []
        batch_images = []
        
        for sample in batch_samples:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": "Has this image been manipulated or synthesized?"}
                ]
            }]
            batch_messages.append(messages)
            batch_images.append(sample["image"])
        
        # Process batch
        batch_texts = []
        all_image_inputs = []
        
        for messages in batch_messages:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(text)
            image_inputs, _ = process_vision_info(messages)
            all_image_inputs.extend(image_inputs)
        
        # Create batch inputs
        inputs = processor(
            text=batch_texts, 
            images=all_image_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        return generated_ids

    def load_model_with_optimization(self, optimization_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load model with specified optimization"""
        config = config or {}
        
        # Model loading arguments
        model_kwargs = {
            "device_map": "auto",
        }
        
        # Apply optimization-specific configurations
        if optimization_type == "baseline":
            model_kwargs["torch_dtype"] = torch.float32
        elif optimization_type == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif optimization_type == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
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
        elif optimization_type == "torch_compile":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif optimization_type == "flash_attention":
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        # Load base model
        load_start = time.time()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_id,
            **model_kwargs
        )
        
        # Apply post-loading optimizations
        if optimization_type == "torch_compile":
            model = torch.compile(model)
        elif optimization_type == "flash_attention" and hasattr(model.config, 'use_flash_attention_2'):
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

        accuracy_metrics = self.calculate_accuracy(model_components, num_samples=50)

        # Measure batch throughput
        batch_metrics = self.measure_batch_throughput(
            model_components,
            batch_sizes=[1, 4, 8, 16]
        )
        
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

        # Measure batch throughput
        batch_metrics = self.measure_batch_throughput(
            model_components,
            batch_sizes=[1, 4, 8, 16]
        )
        
        return {
            "model_size_mb": model_size_mb,
            "inference_time_ms": avg_inference_time,
            "memory_usage_mb": np.mean(memory_measurements) if memory_measurements else 0,
            "throughput_images_per_sec": batch_metrics['max_throughput_images_per_sec'],
            "batch_throughput_metrics": batch_metrics,
            "gpu_utilization_percent": np.mean(gpu_utilizations) if gpu_utilizations else 0,
            "cuda_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
            "accuracy": accuracy_metrics["accuracy"],
            "accuracy_metrics": accuracy_metrics
        }
    
    def _run_inference(self, sample: Dict, model, processor, device, return_text=False):
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

        if return_text:
            # Decode the generated text
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract ground truth from sample if available
            ground_truth = None
            if "item" in sample and "conversations" in sample["item"]:
                # Assuming ground truth is in the assistant's response
                for conv in sample["item"]["conversations"]:
                    if conv.get("from") == "assistant":
                        ground_truth = conv.get("value", "").lower()
                        break
            
            return generated_ids, generated_text, ground_truth
        
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
        
        # Start MLflow run: the tracking starts here 
        mlflow.set_experiment(self.mlflow_experiment_name)
        with mlflow.start_run(nested=True, run_name=f"optimization_{optimization_type}_{self.model_version}") as run:
            
            # Log parameters
            mlflow.log_param("optimization_type", optimization_type)
            mlflow.log_param("model_version", self.model_version)
            mlflow.log_params(config)
            
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
            mlflow.log_metric("accuracy", metrics["accuracy"])
                
            # Create result
            result = OptimizationResult(
                name=f"{optimization_type}_{json.dumps(config)}",
                model_type="qwen2_vl",
                model_version=self.model_version,
                optimization_type=optimization_type,
                model_size_mb=metrics["model_size_mb"],
                inference_time_ms=metrics["inference_time_ms"],
                memory_usage_mb=metrics["memory_usage_mb"],
                accuracy_metrics=["accuracy_metrics"],
                config=config,
                load_time_s=model_components["load_time"],
                throughput_images_per_sec=metrics["throughput_images_per_sec"],
                timestamp=datetime.now().isoformat(),
                gpu_utilization_percent=metrics["gpu_utilization_percent"],
                cuda_memory_allocated_mb=metrics["cuda_memory_allocated_mb"],
                mlflow_run_id=run.info.run_id
            )
            
            self.results.append(result)
            return result
        
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
        report += "| Optimization | Size (MB) | Inference (ms) | Memory (MB) | Throughput | GPU % | Accuracy % | Acc. Drop |\n"
        report += "|-------------|-----------|----------------|-------------|------------|-------|------------|----------|\n"
    
        baseline_accuracy = self.baseline_metrics.accuracy_metrics.get("accuracy", 0.0)

        for result in self.results:

            accuracy = result.accuracy_metrics.get("accuracy", 0.0)
            accuracy_drop = baseline_accuracy - accuracy

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

            report += "\n## Batch Throughput Analysis\n\n"
    
            for result in self.results:
                if "batch_throughput" in result.config:
                    report += f"\n### {result.optimization_type}\n"
                    report += f"Optimal batch size: {result.config.get('optimal_batch_size', 'N/A')}\n"
                    
                    throughput_data = result.config.get('batch_throughput', {})
                    if throughput_data:
                        report += "\n| Batch Size | Throughput (img/s) |\n"
                        report += "|------------|-------------------|\n"
                        for batch_size, throughput in sorted(throughput_data.items()):
                            report += f"| {batch_size.replace('batch_', '')} | {throughput:.2f} |\n"
        
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