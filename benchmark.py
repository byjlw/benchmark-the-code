import logging
import tempfile
from typing import Dict, List
from tqdm import tqdm
from human_eval.data import write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

from model import ModelBenchmark
from utils import format_completion, save_results

def benchmark_model(model: str, problems: Dict, num_samples: int = None) -> Dict:
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
                formatted = format_completion(task_id, completion)
                if formatted:
                    results.append(formatted)
                    successful += 1
                    pbar.set_postfix({"success": f"{successful}/{total}"})
                else:
                    logging.warning(f"Skipping malformed completion for task {task_id}")
            else:
                logging.warning(f"Skipping task {task_id} for {model} due to generation failure")
    
    if successful < total:
        logging.warning(f"Completed {successful}/{total} tasks successfully for {model}")
    else:
        logging.info(f"Successfully completed all {total} tasks for {model}")
    
    try:
        all_completions = []
        completion_map = {c["task_id"]: c["completion"] for c in results}
        
        for task_id in problems.keys():
            if task_id in completion_map:
                all_completions.append({
                    "task_id": task_id,
                    "completion": completion_map[task_id],
                    "prompt": None
                })
            else:
                all_completions.append({
                    "task_id": task_id,
                    "completion": "",
                    "prompt": None
                })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl') as temp:
            write_jsonl(temp.name, all_completions)
            metrics = evaluate_functional_correctness(temp.name)
        
        attempted_pass_rate = (
            metrics["pass@1"] * len(problems) / len(results)
            if results else 0.0
        )
        
        final_results = {
            "completions": results,
            "metrics": metrics,
            "attempted": len(results),
            "total_requested": total,
            "pass_rate": attempted_pass_rate
        }
        
        logging.info(f"Evaluation results for {model}:")
        logging.info(f"Attempted: {final_results['attempted']}/{final_results['total_requested']} problems")
        logging.info(f"Pass rate (of attempted): {final_results['pass_rate']*100:.1f}%")
        logging.info(f"Raw pass rate (including empty): {metrics['pass@1']*100:.1f}%")
        
        return final_results
        
    except Exception as e:
        logging.error(f"Evaluation failed for {model}: {str(e)}")
        logging.error(f"Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"Traceback:\n{traceback.format_exc()}")
        return None
