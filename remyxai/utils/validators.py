import os
import re
import logging
import requests
from typing import List, Tuple, Optional
from huggingface_hub import HfFolder
from remyxai.api.models import fetch_supported_architectures

def get_hf_token() -> Optional[str]:
    """
    Fetches the Hugging Face token from the user's environment.
    Tries environment variable 'HF_TOKEN' first, then the Hugging Face cache.
    """
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        return hf_token
    hf_token = HfFolder.get_token()
    return hf_token

def get_headers(hf_token: Optional[str]):
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    return headers

def validate_model_architecture(
    model_id: str, supported_archs: List[str], hf_token: Optional[str]
) -> Tuple[bool, str]:
    """
    Validates if a model's architecture matches any known architectures from the server.
    """
    try:
        api_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        response = requests.get(api_url, headers=get_headers(hf_token), timeout=10)
        response.raise_for_status()

        config = response.json()
        architectures = config.get("architectures", [])

        if not supported_archs:
            return False, "Supported architectures list is empty."

        for architecture in architectures:
            if architecture in supported_archs:
                return True, f"Model '{model_id}' matches architecture: {architecture}"

        return False, (
            f"Model '{model_id}' does not match any supported architectures: {supported_archs}"
        )

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return False, f"Model '{model_id}' does not exist or is not accessible."
        else:
            return False, f"HTTP error fetching model config for '{model_id}': {e}"
    except requests.exceptions.RequestException as e:
        return False, f"Request error fetching model config for '{model_id}': {e}"
    except KeyError:
        return False, f"Invalid configuration format for '{model_id}'"
    except Exception as e:
        logging.error(f"Unexpected error during architecture validation: {e}")
        return False, f"Unexpected error during validation: {e}"

def validate_model_size(model_id: str, max_size_billion: int = 8) -> Tuple[bool, str]:
    """
    Validates a model's size based on its repository name convention.
    """
    pattern = r"(\d+(\.\d+)?)([BM])"
    match = re.search(pattern, model_id)

    if not match:
        return True, (
            f"Model size for '{model_id}' could not be determined from the name; "
            "assumed valid."
        )

    size_str, _, unit = match.groups()
    size = float(size_str)

    if unit.upper() == "M":
        size /= 1000  # Convert millions to billions if necessary

    if size <= max_size_billion:
        return True, (
            f"Model '{model_id}' size ({size}B) is within the allowed limit of "
            f"{max_size_billion}B."
        )
    else:
        return False, (
            f"Model '{model_id}' size ({size}B) exceeds the allowed limit of "
            f"{max_size_billion}B."
        )

def validate_model(
    model_id: str, supported_archs: List[str], hf_token: Optional[str], max_size_billion: int = 8
) -> Tuple[bool, str]:
    """
    Validates a model based on its architecture and size.
    """
    is_architecture_valid, arch_reason = validate_model_architecture(
        model_id, supported_archs, hf_token
    )
    if not is_architecture_valid:
        return False, arch_reason

    is_size_valid, size_reason = validate_model_size(model_id, max_size_billion)
    if not is_size_valid:
        return False, size_reason

    return True, f"Model '{model_id}' passed validation for architecture and size."

def _validate_models(
    models: List[str], max_size_billion: int = 8
):
    """
    Validates a list of models, raising an exception if any fail validation.
    Automatically fetches the user's HF token from the environment.
    """
    # Fetch the supported architectures once
    supported_archs = fetch_supported_architectures()
    if not supported_archs:
        raise ValueError("Failed to fetch supported architectures from server.")

    # Fetch the HF token
    hf_token = get_hf_token()
    if not hf_token:
        logging.warning("No Hugging Face token found; only public models can be validated.")

    invalid_models = []
    reasons = []

    for model in models:
        is_valid, reason = validate_model(model, supported_archs, hf_token, max_size_billion)
        if not is_valid:
            invalid_models.append(model)
            reasons.append(reason)
            logging.warning(reason)
        else:
            logging.info(reason)

    if invalid_models:
        error_messages = "\n".join(reasons)
        raise ValueError(f"The following models failed validation:\n{error_messages}")
