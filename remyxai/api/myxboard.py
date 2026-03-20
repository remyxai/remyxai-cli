import logging
import requests
import urllib.parse
from typing import Optional
from . import BASE_URL, HEADERS, get_headers, log_api_response


def _resolve_headers(api_key: Optional[str] = None) -> dict:
    """Return headers using explicit key if provided, else module-level default."""
    if api_key:
        return get_headers(api_key)
    return HEADERS


def store_myxboard(
    name: str, models: list, results: dict = None, api_key: Optional[str] = None
) -> dict:
    """Create and store a new MyxBoard on the server."""
    url = f"{BASE_URL}/myxboard/store"
    payload = {"name": name, "models": models, "results": results or None}
    response = requests.post(
        url, json=payload, headers=_resolve_headers(api_key)
    )

    log_api_response(response)

    if response.status_code == 201:
        return response.json()
    else:
        logging.error(f"Failed to create MyxBoard: {response.status_code}")
        return {"error": f"Failed to create MyxBoard: {response.text}"}


def list_myxboards(api_key: Optional[str] = None) -> list:
    """List all MyxBoards from the server."""
    url = f"{BASE_URL}/myxboard/list"
    response = requests.get(url, headers=_resolve_headers(api_key))

    log_api_response(response)

    if response.status_code == 200:
        return response.json().get("message", [])
    else:
        logging.error(f"Failed to fetch MyxBoard list: {response.status_code}")
        return {"error": f"Failed to fetch MyxBoard list: {response.text}"}


def update_myxboard(
    myxboard_id: str,
    models: list,
    results: dict = None,
    from_hf_collection: bool = False,
    hf_collection_name: str = None,
    api_key: Optional[str] = None,
) -> dict:
    """Update an existing MyxBoard on the server."""
    url = f"{BASE_URL}/myxboard/update/{myxboard_id}"
    payload = {
        "models": models,
        "results": results or {},
        "from_hf_collection": from_hf_collection,
        "hf_collection_name": hf_collection_name,
    }
    logging.info(f"PUT request to {url} with payload: {payload}")
    response = requests.put(
        url, json=payload, headers=_resolve_headers(api_key)
    )

    if response.status_code == 200:
        try:
            return response.json()
        except (requests.JSONDecodeError, ValueError) as e:
            logging.error(f"Error decoding JSON response: {e}")
            return {"error": "Invalid JSON response"}
    else:
        logging.error(f"Failed to update MyxBoard: {response.status_code}")
        return {"error": f"Failed to update MyxBoard: {response.text}"}


def delete_myxboard(myxboard_id: str, api_key: Optional[str] = None) -> dict:
    """Delete an existing MyxBoard from the server."""
    url = f"{BASE_URL}/myxboard/delete/{myxboard_id}"
    logging.info(f"DELETE request to {url}")
    response = requests.delete(url, headers=_resolve_headers(api_key))

    if response.status_code == 200:
        try:
            return response.json()
        except (requests.JSONDecodeError, ValueError) as e:
            logging.error(f"Error decoding JSON response: {e}")
            return {"error": "Invalid JSON response"}
    else:
        logging.error(f"Failed to delete MyxBoard: {response.status_code}")
        return {"error": f"Failed to delete MyxBoard: {response.text}"}


def download_myxboard(
    myxboard_name: str, api_key: Optional[str] = None
) -> dict:
    """Download a MyxBoard's results using the name."""
    url = f"{BASE_URL}/myxboard/download/{myxboard_name}"
    logging.info(f"GET request to {url}")
    response = requests.get(url, headers=_resolve_headers(api_key))

    if response.status_code == 200:
        try:
            results = response.json()
            if "message" in results:
                return results["message"]
            else:
                return results
        except (requests.JSONDecodeError, ValueError) as e:
            logging.error(f"Error decoding JSON response: {e}")
            return {"error": "Invalid JSON response"}
    else:
        logging.error(f"Failed to download MyxBoard: {response.status_code}")
        return {"error": f"Failed to download MyxBoard: {response.text}"}
