import argparse
import json
import requests
import time
from datetime import datetime
import tempfile
import logging
import os
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class ModelBenchmark:
    def __init__(self, model: str, base_url: str = "http://localhost:11434/api/generate", timeout: int = 300):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def generate_solution(self, prompt: str, retries: int = 3) -> Optional[str]:
        for attempt in range(retries):
            try:
                response = self.session.post(
                    self.base_url,
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["response"]
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    logging.error(f"Failed to generate solution for {self.model} after {retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )

def format_completion(task_id: str, completion: str) -> Dict:
    """Format a completion in the way expected by human-eval."""
    return {
        "task_id": task_id,
        "completion": completion,
        "prompt": None  # Not needed for evaluation
    }

def benchmark_model(model: str, problems: Dict, num_samples: int = None) -> List[Dict]:
    results = []
    successful = 0
    total = num_samples if num_samples else len(problems)
    
    if num_samples:
        problems = dict(list(problems.items())[:num_samples])
    
    with ModelBenchmark(model) as benchmark:
        pbar = tqdm(problems.items(), desc=f"Testing {model}", total=total)
        for task_id, problem in pbar:
            completion = benchmark.generate_solution(problem["prompt"])
            if completion is not None:
                results.append(format_completion(task_id, completion))
                successful += 1
                pbar.set_postfix({"success": f"{successful}/{total}"})
            else:
                logging.warning(f"Skipping task {task_id} for {model} due to generation failure")
    
    if successful < total:
        logging.warning(f"Completed {successful}/{total} tasks successfully for {model}")
    else:
        logging.info(f"Successfully completed all {total} tasks for {model}")
    
    return results

def save_results(results: Dict, output_dir: Path, model: str) -> bool:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{model}_{timestamp}"
        
        completions_path = output_dir / f"{base_filename}_completions.jsonl"
        write_jsonl(str(completions_path), results["completions"])
        
        metrics_path = output_dir / f"{base_filename}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=2)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save results for {model}: {e}")
        return False

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
    parser.add_argument("--timeout", type=int, default=300,
                      help="Timeout for each API call in seconds (default: 300)")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    
    try:
        problems = read_problems()
    except Exception as e:
        logging.error(f"Failed to read problems: {e}")
        return
    
    logging.info(f"Starting benchmark with models: {args.models}")
    
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
    
    for model, completions in zip(args.models, model_completions):
        if not completions:
            logging.error(f"No completions generated for {model}")
            continue
            
        try:
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl') as temp:
                    write_jsonl(temp.name, completions)
                    metrics = evaluate_functional_correctness(temp.name)
                    
                results = {
                    "completions": completions,
                    "metrics": metrics,
                    "attempted": len(completions),
                    "total_requested": args.samples if args.samples else len(problems),
                    "pass_rate": metrics["pass@1"]
                }
                
                logging.info(f"Evaluation results for {model}:")
                logging.info(f"Attempted: {results['attempted']}/{results['total_requested']} problems")
                logging.info(f"Pass rate: {results['pass_rate']*100:.1f}%")
                
            except Exception as e:
                logging.error(f"Evaluation failed for {model}: {str(e)}")
                logging.error(f"Exception type: {type(e).__name__}")
                import traceback
                logging.error(f"Traceback:\n{traceback.format_exc()}")
                raise
            
            if save_results(results, output_dir, model):
                logging.info(f"Results saved for {model}")
                print(f"\nResults for {model}:")
                print(json.dumps(metrics, indent=2))
            
        except Exception as e:
            logging.error(f"Failed to process results for {model}: {e}")

if __name__ == "__main__":
    main()
