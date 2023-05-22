import os
import shutil
import requests

REMYXAI_API_KEY = os.environ.get("REMYXAI_API_KEY")
if not REMYXAI_API_KEY:
    raise ValueError("REMYXAI_API_KEY not found in environment variables. Please set it with your API key.")

BASE_URL = "https://engine.remyx.ai/api/v1.0/"

HEADERS = {
    "authorization": f"Bearer {REMYXAI_API_KEY}",
}

# Models
def list_models():
    url = f"{BASE_URL}model/list"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def get_model_summary(model_name):
    url = f"{BASE_URL}model/summary/{model_name}"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def delete_model(model_name: str):
    url = f"{BASE_URL}model/delete/{model_name}"
    response = requests.post(url, headers=HEADERS)
    return response.json()

def download_model(model_name: str, model_format: str):
    url = f"{BASE_URL}model/download/{model_name}/{model_format}"
    response = requests.post(url, headers=HEADERS, stream=True)
    with open(f"{model_name}.zip", "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    return f"The file {model_name}.zip was saved successfully"

# Engines
def train_classifier(model_name: str, labels: list, model_selector: str):
    url = f"{BASE_URL}task/classify/{model_name}/{','.join(labels)}/{model_selector}"
    response = requests.post(url, headers=HEADERS)
    return response.json()

# User
def get_user_profile():
    url = f"{BASE_URL}user"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def get_user_credits():
    url = f"{BASE_URL}user/credits"
    response = requests.get(url, headers=HEADERS)
    return response.json()

