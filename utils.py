import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
import json
from human_eval.data import write_jsonl

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
    if not completion.strip().startswith("def"):
        logging.warning(f"Completion for {task_id} doesn't start with 'def'")
        return None
        
    return {
        "task_id": task_id,
        "completion": completion,
        "prompt": None
    }

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
