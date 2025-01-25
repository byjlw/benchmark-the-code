import logging
import requests
import time
from typing import Optional

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
                completion = response.json()["response"]
                
                if "def" in completion:
                    completion = completion[completion.find("def"):]
                    logging.debug(f"Generated completion:\n{completion}")
                    return completion
                else:
                    logging.warning(f"No function definition found in completion:\n{completion}")
                    return None
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    logging.error(f"Failed to generate solution for {self.model} after {retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)
