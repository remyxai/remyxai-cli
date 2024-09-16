import os
import logging

REMYXAI_API_KEY = os.getenv("REMYXAI_API_KEY")
if not REMYXAI_API_KEY:
    raise ValueError("REMYXAI_API_KEY not found in environment variables. Please set it with your API key.")

BASE_URL = "https://engine.remyx.ai/api/v1.0/"

HEADERS = {
    "authorization": f"Bearer {REMYXAI_API_KEY}",
}

def log_api_response(response):
    """Log the response from the API."""
    if response.status_code == 200:
        logging.info(f"API call successful: {response.url}")
    else:
        logging.error(f"API call failed: {response.url}, Status: {response.status_code}, Response: {response.text}")

