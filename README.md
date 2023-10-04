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
Currently, the classify engine API is supported. Additional functionality for model management and user data is available.

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
* python command:
```python
from remyxai.api import train_classifier

model_name = "<your-model-name>"
labels = ["comma", "separated", "labels"]
model_size = 1
print(train_classifier(model_name, labels, model_size))
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
*New!* Label images locally:
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

