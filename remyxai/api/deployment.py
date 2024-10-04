import os
import shutil
import tempfile
import subprocess
import requests
from . import BASE_URL, HEADERS, log_api_response


def download_deployment_package(model_name, output_path):
    url = f"{BASE_URL}deployment/download/{model_name}"
    response = requests.get(url, headers=HEADERS, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        return response
    else:
        return None


def deploy_model(model_name, action="up"):
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_dir = os.path.join(tmpdirname, model_name)
        compose_file_path = os.path.join(model_dir, "docker-compose.yml")
        zip_path = os.path.join(tmpdirname, f"{model_name}_deployment_package.zip")

        if action == "up":
            if download_deployment_package(model_name, zip_path):
                os.makedirs(model_dir, exist_ok=True)
                subprocess.run(["unzip", "-o", zip_path, "-d", model_dir], check=True)
                if not os.path.exists(compose_file_path):
                    with open(compose_file_path, "w") as f:
                        f.write(
                            f"""
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
                        """
                        )
                os.chdir(model_dir)
                subprocess.run(["docker", "compose", "up", "--build", "-d"], check=True)
        elif action == "down":
            if os.path.exists(compose_file_path):
                os.chdir(model_dir)
                subprocess.run(["docker", "compose", "down"], check=True)
