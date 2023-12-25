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

    if response.status_code == 200:
        # Extract filename from Content-Disposition header if available
        content_disposition = response.headers.get('content-disposition')
        if content_disposition:
            filename = content_disposition.split('filename=')[1]
            filename = filename.strip("\"'")
        else:
            # Fallback to default naming convention
            filename = f"{model_name}.zip"

        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return f"The file {filename} was saved successfully"
    else:
        return f"Failed to download the model. Status Code: {response.status_code}"


# Engines
def train_classifier(model_name: str, labels: list, model_selector: str, hf_dataset=None):
    url = f"{BASE_URL}task/classify/{model_name}/{','.join(labels)}/{model_selector}"
    if hf_dataset:
         params = {"hf_dataset": hf_dataset}
         response = requests.post(url, headers=HEADERS, params=params)
         return response.json()

    response = requests.post(url, headers=HEADERS)
    return response.json()

def train_detector(model_name: str, labels: list, model_selector: str, hf_dataset=None):
    url = f"{BASE_URL}task/detect/{model_name}/{','.join(labels)}/{model_selector}"
    if hf_dataset:
         params = {"hf_dataset": hf_dataset}
         response = requests.post(url, headers=HEADERS, params=params)
         return response.json()
    response = requests.post(url, headers=HEADERS)
    return response.json()

def train_generator(model_name: str, hf_dataset: str):
    url = f"{BASE_URL}task/generate/{model_name}"
    params = {"hf_dataset": hf_dataset}
    response = requests.post(url, headers=HEADERS, params=params)
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

