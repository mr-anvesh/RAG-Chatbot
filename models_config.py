"""
Configuration file for available models.
Add or remove models by updating the AVAILABLE_MODELS dictionary.
"""

AVAILABLE_MODELS = {
    "Gemma-3B (Free)": {
        "id": "google/gemma-3-4b-it:free",
        "context_length": 8000,
        "description": "Google's latest open-source model, optimized for instruction following"
    },
    "Deepseek-Zero (Free)": {
        "id": "deepseek/deepseek-r1-zero:free",
        "context_length": 8000,
        "description": "A powerful open-source model, free to use"
    },
    "Mistral-7B (Free)": {
        "id": "mistralai/mistral-7b-instruct:free",
        "context_length": 4000,
        "description": "Efficient open-source model with good performance"
    }
} 
