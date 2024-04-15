# Remyx AI command-line client

## Installation
To install the Remyx AI CLI in Python virtual environment, run:

```
pip install remyxai
```

## Token authentication
Remyx AI API requires authentication token, which can be obtained on this page: https://engine.remyx.ai/account

Provide api key to the CLI through an environment variable `REMYXAI_API_KEY`.
```
export REMYXAI_API_KEY=<your-key-here>
```

## Usage
Quickly get started with the following examples:

### Model
List all models:
* cli command:
```bash
$ remyxai model list
```
* python command:
```python
from remyxai.api import list_models
print(list_models())
```

Get the summary of a model:
* cli command:
```bash
$ remyxai model summarize --model_name=<your-model-name>
```
* python command:
```python
from remyxai.api import get_model_summary
print(get_model_summary(model_name))
```

Delete a model by name:
* cli command:
```bash
$ remyxai model delete --model_name=<your-model-name>
```
* python command:
```python
from remyxai.api import delete_model

model_name = "<your-model-name>"
print(delete_model(model_name))
```

Download and convert a model:
* cli command:
```bash
# possible model formats are "blob", "onnx", or "tflite"
$ remyxai model download --model_name=<your-model-name> --model_format="onnx"
```
* python command:
```python
from remyxai.api import download_model 

model_name = "<your-model-name>"
model_format = "onnx"
print(download_model(model_name, model_format))
```

### Tasks
Train an image classifier:
* cli command:
```bash
$ remyxai classify --model_name=<your-model-name> --labels="comma,separated,labels" --model_size=<int between 1-5>
```

add the optional `--hf_dataset` if you want to train with your own image dataset on 🤗. [See the docs](https://huggingface.co/docs/datasets/v2.14.5/image_dataset#imagefolder) for more details

* python command:
```python
from remyxai.api import train_classifier

model_name = "<your-model-name>"
labels = ["comma", "separated", "labels"]
model_size = 3 # use 1 for microcontrollers

# Optional HF dataset
hf_dataset = "your/hf-dataset"

print(train_classifier(model_name, labels, model_size, hf_dataset))
```

Train an object detector:
* cli command:
```bash
$ remyxai detect --model_name=<your-model-name> --labels="comma,separated,labels" --model_size=<int between 1-5>
```

add the optional `--hf_dataset` if you want to train with your own image dataset on 🤗. [See the docs](https://huggingface.co/docs/datasets/v2.14.5/image_dataset#object-detection) for more details

* python command:
```python
from remyxai.api import train_detector

model_name = "<your-model-name>"
labels = ["comma", "separated", "labels"]
model_size = 3

# Optional HF dataset
hf_dataset = "your/hf-dataset"
print(train_detector(model_name, labels, model_size, hf_dataset))
```

Train a text generator:
* cli command:
```bash
$ remyxai generate --model_name=<your-model-name> --hf_dataset=<your/hf-dataset>
```

Your Huggingface dataset should have two columns with naming conventions like:
* "question", "response"
* "question", "answer"
* "input", "output"
* "prompt", "response"

* python command:
```python
from remyxai.api import train_generator

model_name = "<your-model-name>"
hf_dataset = "your/hf-dataset"

print(train_generator(model_name, hf_dataset))
```
### Deploy 
Launch a [Triton Server](https://developer.nvidia.com/triton-inference-server) containerized deployment for your model. Currently supported for `generate` models. More model types support coming soon!

#### System requirements
Please make sure you have Docker, Docker Compose, and the NVIDIA Container Toolkit are installed. 
* [Docker installation](https://docs.docker.com/engine/install/ubuntu/)
* [Docker Compose installation](https://docs.docker.com/compose/install/linux/)
* [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Deploy a model with:
* cli command:
```bash
# Bring up
remyxai deploy --model_name="<your-model-name>"

# Bring down
remyxai deploy down --model_name="<your-model-name>"
```

* python command:
```python
from remyxai.api import deploy_model

model_name = "<your-model-name>"

deploy_model(model_name, action='up') # action can be "up" or "down"
```

And you can run inference with:
* cli command:
```bash
remyxai infer --model_name="<your-model-name>" --prompt="Your prompt here"
```

* python command:
```python
from remyxai.api import run_inference

model_name = "<your-model-name>"
prompt="Your prompt here"

result, time_elapsed = run_inference(model_name, prompt, server_url="localhost:8000", model_version="1")
print(result)
```

### User

Get user profile info:
* cli command:
```bash
$ remyxai user profile
```
* python command:
```python
from remyxai.api import get_user_profile

print(get_user_profile())
```


Get user credit/subscription info:
* cli command:
```bash
$ remyxai user credits
```
* python command:
```python
from remyxai.api import get_user_credits

print(get_user_credits())
```

### Utils
Label images locally:
* cli command:
```bash
$ remyxai utils label --labels="comma,separated,labels" --image_dir="/path/to/image/dir"
```

* python command:
```python
from remyxai.utils import labeler
model_name = "<your-model-name>"
labels = ["comma", "separated", "labels"]
image_dir = "/path/to/image/dir"
print(labeler(labels, image_dir, model_name))
```

