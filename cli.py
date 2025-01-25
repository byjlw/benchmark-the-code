import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List

from benchmark import benchmark_model
from utils import setup_logging
from human_eval.data import read_problems
from concurrent.futures import ThreadPoolExecutor

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
    
    for model, results in zip(args.models, model_completions):
        if results:
            print(f"\nResults for {model}:")
            print(json.dumps(results["metrics"], indent=2))

if __name__ == "__main__":
    main()
