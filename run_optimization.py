# seprate script that runs the optimization tests pipeline
# This script orchestrates the optimization pipeline defined in optimization_pipeline_integrated : it effectively acts as an entrypoint
# When this script is called it creates a parent MFLOW run and initializes the optimizer 
# Then it calls baseline_measure_performance() and logs the baseline performance metrics 
# after baseline tests are completed run_optimization_suite is called which runs the optimization tests concurently 
# for each optimization test a nested Mlflow run is created(as a child of the parent run)
#Results of the test are saved and the artifacts are logged to MLFLOW

"""Parent Run: optimization_suite_v1.0_20240101_120000
MLflow Server
├── Optimization Run (v1.0) - 2024-01-15
│   ├── Baseline: 150ms inference, 2GB memory
│   ├── BF16: 100ms inference, 1.5GB memory     ← Best for speed
│   ├── INT8: 120ms inference, 0.8GB memory     ← Best for memory
│   └── Artifacts: optimization_report.md, deployment_recommendation.json
│
├── Optimization Run (v1.1) - 2024-01-20
│   ├── Baseline: 140ms inference, 2GB memory
│   ├── BF16: 95ms inference, 1.5GB memory
│   └── ...
│
└── Optimization Run (v2.0) - 2024-02-01
    └── ... all optimization results
"""


import argparse
import mlflow
import json
from datetime import datetime
from optimization_pipeline_integrated import QwenModelOptimizer # IMPORTS THE QWENMODELOPTIMIZER CLASS FROM THE OPTIMIZATION PIPELINE IMPLEMENTATION FILE 

def main():
    parser = argparse.ArgumentParser(description="Run model optimization evaluation")
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-3B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--experiment-name", default="Qwen_Optimization")
    parser.add_argument("--mlflow-uri", default=None)
    parser.add_argument("--tags", nargs='+', default=[], help="Additional tags for MLflow")
    
    args = parser.parse_args()
    
    # Set up MLflow
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    
    # Define optimizations
    optimizations = [
        {"type": "bf16"},
        {"type": "fp16"},
        {"type": "quantization_8bit"},
        {"type": "quantization_4bit"},
        {"type": "torch_compile"},
        {"type": "flash_attention"}
    ]
    
    # Starts the parent run
    with mlflow.start_run(run_name=f"optimization_suite_{args.model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as parent_run:
        # Log parameters
        mlflow.log_param("model_version", args.model_version)
        mlflow.log_param("base_model", args.base_model)
        mlflow.log_param("checkpoint_path", args.checkpoint)
        mlflow.log_param("test_data_path", args.test_data)
        mlflow.log_param("num_optimizations", len(optimizations))
        
        # Log tags
        mlflow.set_tags({
            "optimization_type": "full_suite",
            "model_family": "qwen2_vl",
            **{f"custom_{i}": tag for i, tag in enumerate(args.tags)}
        })
        
        # Initialize the optimizer
        optimizer = QwenModelOptimizer(
            base_model_id=args.base_model,
            checkpoint_path=args.checkpoint,
            test_dataset_path=args.test_data,
            model_version=args.model_version,
            mlflow_experiment_name=args.experiment_name
        )
        
        # Run the baseline tests with no optimizatiosn 
        print("Running baseline measurement...")
        baseline = optimizer.baseline_measure_performance()
        mlflow.log_metric("baseline_inference_time", baseline.inference_time_ms)
        mlflow.log_metric("baseline_memory_usage", baseline.memory_usage_mb)
        
        # Run the optimization suite : the co
        print("Running optimization suite...")
        results = optimizer.run_optimization_suite(optimizations)
        
        # Log summary metrics
        mlflow.log_metric("best_inference_time", 
                         min(r.inference_time_ms for r in results))
        mlflow.log_metric("best_memory_usage", 
                         min(r.memory_usage_mb for r in results))
        mlflow.log_metric("best_throughput", 
                         max(r.throughput_images_per_sec for r in results))
        
        # Calculate improvements
        best_result = min(results, key=lambda r: r.inference_time_ms)
        improvement_percent = (baseline.inference_time_ms - best_result.inference_time_ms) / baseline.inference_time_ms * 100
        mlflow.log_metric("inference_improvement_percent", improvement_percent)
        
        # Log artifacts
        mlflow.log_artifact(f"{optimizer.results_dir}/deployment_recommendation.json")
        mlflow.log_artifact(f"{optimizer.results_dir}/optimization_report.md")
        mlflow.log_artifact(f"{optimizer.results_dir}/optimization_results.json")
        
        # Log best configuration as a separate artifact
        best_config = {
            "optimization_type": best_result.optimization_type,
            "config": best_result.config,
            "metrics": {
                "inference_time_ms": best_result.inference_time_ms,
                "memory_usage_mb": best_result.memory_usage_mb,
                "throughput_images_per_sec": best_result.throughput_images_per_sec
            },
            "improvement_over_baseline": f"{improvement_percent:.1f}%"
        }
        
        with open(f"{optimizer.results_dir}/best_configuration.json", "w") as f:
            json.dump(best_config, f, indent=2)
        mlflow.log_artifact(f"{optimizer.results_dir}/best_configuration.json")
        
        print(f"\nOptimization complete!")
        print(f"Results saved to: {optimizer.results_dir}")
        print(f"MLflow run ID: {parent_run.info.run_id}")
        print(f"Best optimization: {best_result.optimization_type}")
        print(f"Improvement over baseline: {improvement_percent:.1f}%")

if __name__ == "__main__":
    main()