import requests
from config import HF_API_KEY
from colorama import Fore, Style, init

init(autoreset=True)

DEFAULT_MODEL = "google/pegasus-xsum"

def build_api_url(model_name):
    # New router-based inference endpoint recommended by Hugging Face
    base_url = "https://router.huggingface.co/hf-inference/models/"
    return f"{base_url}{model_name}"

def query(payload, model_name=DEFAULT_MODEL):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    api_url = build_api_url(model_name)
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        print(Fore.RED + f"Request error: {e}")
        return None

    if not response.ok:
        print(Fore.RED + f"HTTP {response.status_code} error:\n{response.text}")
        return None

    try:
        return response.json()
    except ValueError:
        # JSON decode failed — print the raw response for debugging
        print(Fore.RED + "Failed to decode JSON response. Raw response text:")
        print(response.text)
        return None
def summarize_text(text, model_name=DEFAULT_MODEL):
    payload = {"inputs": text}
    response = query(payload, model_name=model_name)
    if response is None:
        return None

    if isinstance(response, dict) and response.get("error"):
        print(Fore.RED + "Error: " + str(response.get("error")))
        return None

    # Some models return a list of results, others return a dict.
    try:
        if isinstance(response, list):
            summary = response[0].get('summary_text') or response[0].get('generated_text')
        elif isinstance(response, dict):
            # Try common keys
            summary = response.get('summary_text') or response.get('generated_text') or None
        else:
            summary = None
    except Exception:
        summary = None

    if not summary:
        print(Fore.YELLOW + "Warning: Could not find summary text in response.")
        print("Full response:", response)
        return None

    return summary
# Note: API key is read from `config.py` at import time. Do not overwrite it here.

if __name__ == "__main__":
    sample_text = (
        "The Hugging Face Transformers library provides a simple and efficient way to use "
        "pre-trained models for natural language processing tasks. It supports a wide range "
        "of models including BERT, GPT-2, T5, and more. With its easy-to-use API, developers "
        "can quickly integrate state-of-the-art NLP capabilities into their applications."
    )
    
    print(Fore.CYAN + "Original Text:\n" + Style.RESET_ALL + sample_text + "\n")
    
    summary = summarize_text(sample_text)
    
    if summary:
        print(Fore.GREEN + "Summary:\n" + Style.RESET_ALL + summary)




