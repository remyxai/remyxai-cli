import os
import logging

REMYXAI_API_KEY = os.getenv("REMYXAI_API_KEY")
if not REMYXAI_API_KEY:
    logging.error("REMYXAI_API_KEY not found in environment variables.")
    raise ValueError("REMYXAI_API_KEY not found. Please set it with your API key.")
else:
    logging.info(f"Using API Key: {REMYXAI_API_KEY}")  # Log the key for debugging (only in dev)

BASE_URL = "https://engine.remyx.ai/api/v1.0"

HEADERS = {
    "Authorization": f"Bearer {REMYXAI_API_KEY}",
    "Content-Type": "application/json"
}

def log_api_response(response):
    """Log the response from the API based on the status code."""
    if response.status_code in [200, 201]:
        logging.debug(f"API call successful: {response.url}, Status: {response.status_code}")
    else:
        logging.error(f"API call failed: {response.url}, Status: {response.status_code}, Response: {response.text}")

