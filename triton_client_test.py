# triton_client_test.py
#This file acts as a client for the test.json file and the triton inference server
#It extracts the the image paths and its corresponding prompt/s (some prompts are lists) and then sends request to the triton infernce server 
# results are saved to a csv file for later reference.
import json
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import time
import os
from tqdm import tqdm
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Test Triton Inference Server performance")
    parser.add_argument("--url", default="localhost:8000", help="Triton server URL")
    parser.add_argument("--model", default="qwen_vl_fine_tuned", help="Model name")
    parser.add_argument("--test-file", default="test.json", help="Test JSON file")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to test")
    args = parser.parse_args()

    # Initialize Triton client
    print(f"Connecting to Triton server at {args.url}")
    try:
        client = httpclient.InferenceServerClient(url=args.url)
        if not client.is_server_ready():
            print("ERROR: Triton server is not ready")
            return
        if not client.is_model_ready(args.model):
            print(f"ERROR: Model {args.model} is not ready")
            return
    except Exception as e:
        print(f"ERROR: Failed to connect to Triton server: {e}")
        return

    # the test.json file is opened ans then the data gets loaded
    print(f"Loading test data from {args.test_file}")
    try:
        with open(args.test_file, "r") as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load test file: {e}")
        return

    # Process test samples
    # the samples are extracted from the test.json file : I pre-formatted this file so that the inputs are in the same format
    # that the model expects as input
    test_samples = []
    for item in test_data:
        user_msg = item["conversations"][0]["value"]
        
        # Extract image path
        if "<|vision_start|>" in user_msg and "<|vision_end|>" in user_msg:
            image_path = user_msg.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        else:
            image_path = user_msg
            
        # Get ground truth
        gt_text = item["conversations"][1]["value"]
        if "has not been manipulated" in gt_text.lower():
            gt_label = "real"
        elif "has been manipulated" in gt_text.lower():
            gt_label = "fake"
        else:
            gt_label = "unknown"
            
        test_samples.append({
            "image_path": image_path,
            "ground_truth_text": gt_text,
            "ground_truth_label": gt_label
        })
    
    # Verify the paths to the images  and filter valid samples( i belive all are valid  but just as a good practice)
    valid_samples = []
    for sample in test_samples:
        path = sample["image_path"]
        if os.path.exists(path):
            valid_samples.append(sample)
        else:
            # Try with adjusted path
            base_path = os.path.dirname(os.path.dirname(args.test_file))
            alt_path = os.path.join(base_path, path.lstrip("./"))
            if os.path.exists(alt_path):
                sample["image_path"] = alt_path
                valid_samples.append(sample)
                
    print(f"Found {len(valid_samples)} valid samples out of {len(test_samples)} total samples")
    
    # Limit the number of the samples if needed
    benchmark_samples = valid_samples[:args.num_samples]
    print(f"Using {len(benchmark_samples)} samples for benchmarking")
    
    # Run inference tests
    latencies = []
    results = []
    
    print("Running inference tests...")
    for sample in tqdm(benchmark_samples):
        try:
            # Load image
            image = Image.open(sample["image_path"]).convert("RGB")
            image_np = np.array(image)
            
            # Prepare inputs for Triton
            inputs = []
            
            # Image input
            image_input = httpclient.InferInput("INPUT_IMAGE", image_np.shape, "UINT8")
            image_input.set_data_from_numpy(image_np)
            inputs.append(image_input)
            
            # Text input
            prompt = "Is this image manipulated or synthesized?"
            text_input = httpclient.InferInput("PROMPT", [1], "BYTES")
            text_input.set_data_from_numpy(np.array([prompt.encode()], dtype=np.object_))
            inputs.append(text_input)
            
            # Request outputs
            outputs = [
                httpclient.InferRequestedOutput("prediction"),
                httpclient.InferRequestedOutput("label"),
                httpclient.InferRequestedOutput("reasoning")
            ]
            
            # Run inference and time it
            start_time = time.time()
            response = client.infer(
                model_name=args.model,
                inputs=inputs,
                outputs=outputs
            )
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # ms
            latencies.append(latency)
            
            # Process results
            prediction = response.as_numpy("prediction")[0].decode()
            label = response.as_numpy("label")[0].decode()
            reasoning = response.as_numpy("reasoning")[0].decode()
            
            results.append({
                "image_path": sample["image_path"],
                "ground_truth_label": sample["ground_truth_label"],
                "predicted_label": label,
                "prediction_text": prediction,
                "reasoning": reasoning,
                "latency_ms": latency,
                "is_correct": sample["ground_truth_label"] == label
            })
            
        except Exception as e:
            print(f"Error processing {sample['image_path']}: {e}")
    
    # Calculate performance metrics
    if results:
        # Summary stats
        latencies = [r["latency_ms"] for r in results]
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        correct_count = sum(r["is_correct"] for r in results)
        accuracy = correct_count / len(results)
        
        # Print results
        print("\n===== PERFORMANCE METRICS =====")
        print(f"Average latency: {avg_latency:.2f} ms")
        print(f"Median latency: {median_latency:.2f} ms")
        print(f"95th percentile: {p95_latency:.2f} ms")
        print(f"99th percentile: {p99_latency:.2f} ms")
        print(f"Throughput: {1000/avg_latency:.2f} images/second")
        print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
        
        # Class-specific metrics
        classes = ["real", "fake", "unknown"]
        for cls in classes:
            cls_samples = [r for r in results if r["ground_truth_label"] == cls]
            if cls_samples:
                cls_correct = sum(r["is_correct"] for r in cls_samples)
                cls_accuracy = cls_correct / len(cls_samples)
                print(f"{cls.capitalize()} accuracy: {cls_accuracy:.2%} ({cls_correct}/{len(cls_samples)})")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv("triton_performance_results.csv", index=False)
        print("\nDetailed results saved to triton_performance_results.csv")
        
        # Plot latency distribution
        plt.figure(figsize=(10, 6))
        plt.hist(latencies, bins=20, alpha=0.7)
        plt.axvline(median_latency, color='r', linestyle='--', label=f'Median: {median_latency:.2f} ms')
        plt.axvline(p95_latency, color='g', linestyle='--', label=f'95th: {p95_latency:.2f} ms')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Inference Latency Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('latency_distribution.png')
        print("Latency distribution plot saved to latency_distribution.png")

if __name__ == "__main__":
    main()