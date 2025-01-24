import argparse
import json
import requests
import time
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class ModelBenchmark:
    def __init__(self, model: str, base_url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.base_url = base_url

    def generate_solution(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.base_url,
                json={"model": self.model, "prompt": prompt, "stream": False}
            )
            return response.json()["response"]
        except Exception as e:
            print(f"Error generating solution for {self.model}: {e}")
            return ""

def benchmark_model(model: str, problems: Dict, num_samples: int = None) -> List[Dict]:
    benchmark = ModelBenchmark(model)
    results = []
    
    if num_samples:
        problems = dict(list(problems.items())[:num_samples])
    
    for task_id, problem in tqdm(problems.items(), desc=f"Testing {model}"):
        completion = benchmark.generate_solution(problem["prompt"])
        results.append({
            "task_id": task_id,
            "completion": completion
        })
    
    return results

def save_results(results: Dict, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    completions_path = output_dir / f"completions_{timestamp}.jsonl"
    write_jsonl(str(completions_path), results["completions"])
    
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs using Human-Eval")
    parser.add_argument("--models", nargs="+", required=True,
                      help="List of model names to benchmark")
    parser.add_argument("--samples", type=int,
                      help="Number of problems to test (default: all)")
    parser.add_argument("--output-dir", type=str, default="results",
                      help="Directory to save results (default: ./results)")
    parser.add_argument("--parallel", action="store_true",
                      help="Run models in parallel")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    problems = read_problems()
    
    if args.parallel:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(benchmark_model, model, problems, args.samples)
                for model in args.models
            ]
            model_completions = [f.result() for f in futures]
    else:
        model_completions = [
            benchmark_model(model, problems, args.samples)
            for model in args.models
        ]
    
    results = {}
    for model, completions in zip(args.models, model_completions):
        metrics = evaluate_functional_correctness(completions)
        results[model] = {
            "completions": completions,
            "metrics": metrics
        }
        save_results(results[model], output_dir / model)
        
        print(f"\nResults for {model}:")
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()