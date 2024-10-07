import requests
from . import BASE_URL, HEADERS, log_api_response


def get_user_profile():
    url = f"{BASE_URL}user"
    response = requests.get(url, headers=HEADERS)
    return response.json()


def get_user_credits():
    url = f"{BASE_URL}user/credits"
    response = requests.get(url, headers=HEADERS)
    return response.json()
