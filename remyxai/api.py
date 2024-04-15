import os
import time
import shutil
import requests
import tempfile
import subprocess
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

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

# Deployments
def download_deployment_package(model_name, output_path):
    """Download the deployment package for a specified model."""
    url = f"{BASE_URL}deployment/download/{model_name}"
    response = requests.get(url, headers=HEADERS, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Deployment package downloaded successfully: {output_path}")
        return response  # Return response for further processing if needed
    else:
        print(f"Failed to download deployment package: {response.status_code}")
        print(response.json())  # Assuming the error message is in JSON format
        return None

def deploy_model(model_name, action='up'):
    """Deploy or tear down a model using Docker Compose based on the action."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_dir = os.path.join(tmpdirname, model_name)  # This is where we expect to build the Docker image.
        compose_file_path = os.path.join(model_dir, 'docker-compose.yml')
        zip_path = os.path.join(tmpdirname, f"{model_name}_deployment_package.zip")

        if action == 'up':
            if download_deployment_package(model_name, zip_path):
                # Unzip the package directly into the model_dir to ensure the structure is correct
                os.makedirs(model_dir, exist_ok=True)  # Make sure the target directory exists
                subprocess.run(['unzip', '-o', zip_path, '-d', model_dir], check=True)

                # Check the contents just to verify
                print("Unzipped files:", os.listdir(model_dir))  # For debugging

                # Generate Docker Compose YAML if it does not exist
                if not os.path.exists(compose_file_path):
                    with open(compose_file_path, 'w') as f:
                        f.write(f"""
version: '3.8'
services:
  tritonserver:
    build:
      context: ./
      dockerfile: Dockerfile
    image: {model_name}:latest
    container_name: {model_name}_triton_server
    runtime: nvidia
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    shm_size: 24G
    restart: unless-stopped
                        """)

                # Deploy using Docker Compose
                os.chdir(model_dir)  # Change to the directory where the Dockerfile and docker-compose.yml are
                subprocess.run(['docker', 'compose', 'up', '--build', '-d'], check=True)
                print("Deployment successful")
        elif action == 'down':
            if os.path.exists(compose_file_path):
                os.chdir(model_dir)  # Ensure commands run in the correct directory
                subprocess.run(['docker', 'compose', 'down'], check=True)
                print("Service has been successfully taken down.")
            else:
                print("Error: Deployment not found.")

# Infer
def run_inference(model_name, prompt, server_url="localhost:8000", model_version="1"):
    triton_client = InferenceServerClient(url=server_url, verbose=False)

    # Convert the prompt to bytes and wrap in a numpy array
    prompt_np = np.array([prompt.encode('utf-8')], dtype=object)

    # Create input for PROMPT.
    prompt_in = InferInput(name="PROMPT", shape=[1], datatype="BYTES")
    prompt_in.set_data_from_numpy(prompt_np, binary_data=True)

    results_out = InferRequestedOutput(name="RESULTS", binary_data=False)

    start_time = time.time()
    response = triton_client.infer(model_name=model_name,
                                   model_version=model_version,
                                   inputs=[prompt_in],
                                   outputs=[results_out])

    elapsed_time = time.time() - start_time
    results = response.get_response()["outputs"][0]["data"][0]
    return results, elapsed_time


# User
def get_user_profile():
    url = f"{BASE_URL}user"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def get_user_credits():
    url = f"{BASE_URL}user/credits"
    response = requests.get(url, headers=HEADERS)
    return response.json()

