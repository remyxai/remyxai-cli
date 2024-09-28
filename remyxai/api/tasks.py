import requests
from . import BASE_URL, HEADERS, log_api_response

def get_job_status(job_id: str) -> dict:
    """
    Wraps the API call to retrieve the job status of a running task.
    """
    url = f"{BASE_URL}/api/v1.0/task/job-status/{job_id}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error retrieving job status: {response.status_code}, {response.text}")

def run_myxmatch(name: str, prompt: str, models: list):
    url = f"{BASE_URL}api/v1.0/task/myxmatch/{name}/{prompt}/{','.join(models)}"
    response = requests.post(url, headers=HEADERS)
    
    if response.status_code == 202:
        return response.json()
    else:
        return {"error": f"Failed to create task: {response.json().get('message', 'Unknown error')}"}

def train_classifier(model_name: str, labels: list, model_selector: str, hf_dataset=None):
    url = f"{BASE_URL}task/classify/{model_name}/{','.join(labels)}/{model_selector}"
    params = {"hf_dataset": hf_dataset} if hf_dataset else None
    response = requests.post(url, headers=HEADERS, params=params)
    return response.json()

def train_detector(model_name: str, labels: list, model_selector: str, hf_dataset=None):
    url = f"{BASE_URL}task/detect/{model_name}/{','.join(labels)}/{model_selector}"
    params = {"hf_dataset": hf_dataset} if hf_dataset else None
    response = requests.post(url, headers=HEADERS, params=params)
    return response.json()

def train_generator(model_name: str, hf_dataset: str):
    url = f"{BASE_URL}task/generate/{model_name}"
    params = {"hf_dataset": hf_dataset}
    response = requests.post(url, headers=HEADERS, params=params)
    return response.json()

