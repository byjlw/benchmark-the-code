# benchmark-the-code

Code benchmarking tool for LLMs using OpenAI's HumanEval dataset.

## Installation

```bash
git clone https://github.com/byjlw/benchmark-the-code.git
cd benchmark-the-code

python3 -m venv .venv
souce .venv/bin/activate

pip install -r requirements.txt
```

Requires Ollama if using Ollama models.

## Usage

```bash
# Basic usage
python cli.py --models codellama:7b codellama:13b

# With options
python cli.py \
    --models codellama:7b codellama:13b \
    --samples 10 \
    --parallel \
    --output-dir results \
    --timeout 300
```

## Options

- `--models`: List of model names to benchmark
- `--samples`: Number of problems to test (default: all)
- `--output-dir`: Directory to save results (default: ./results)
- `--parallel`: Run models in parallel
- `--timeout`: Timeout for each API call in seconds (default: 300)

## Output Structure

```
results/
  model_name/
    completions_TIMESTAMP.jsonl  # Raw model outputs
    metrics_TIMESTAMP.json       # Evaluation metrics
```

## Sample Output

```json
{
  "codellama:7b": {
    "pass@1": 0.328,
    "pass@10": 0.421,
    "pass@100": 0.495
  }
}
```

## Credits

- HumanEval dataset: OpenAI
- Evaluation framework: https://github.com/openai/human-eval
