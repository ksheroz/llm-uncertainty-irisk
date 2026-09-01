"""LLM sample generation via local Ollama."""
import requests
from typing import List

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:4b"


def generate_samples(
    question: str,
    n_samples: int = 10,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
    max_tokens: int = 100,
    system_prompt: str = "Answer the question concisely. Give only the answer, no explanation.",
) -> List[str]:
    """Generate n_samples responses from a local Ollama model."""
    prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"
    generations = []

    for _ in range(n_samples):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }, timeout=120)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            text = text.split("\n")[0].strip()
            generations.append(text)
        except Exception as e:
            print(f"  Generation failed: {e}")

    return [g for g in generations if g]
