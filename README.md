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

### Outrider — weekly arXiv → draft PR for your repo

[Outrider](https://github.com/remyxai/outrider) is a GitHub Action that, on a weekly schedule, finds the most implementable recent paper for your repository and opens a draft PR wiring it into a real call site. `remyxai outrider` sets it up for you.

There are two ways to install it — both end with the same Action running in your repo; they differ in **who provisions it**:

|  | `outrider init` | `outrider setup-local` |
|---|---|---|
| **Best for** | Most users | Teams that can't grant a third-party GitHub App yet (e.g. a pending security review) |
| **How it works** | The **Remyx GitHub App** provisions it server-side; PRs are authored by `remyx-ai[bot]` | The CLI uses **your own `gh`** to provision it; PRs authored by `github-actions[bot]` |
| **You provide** | The Remyx App installed on the repo (the command links you to it) | An authenticated `gh` with admin on the repo |

Either way you'll need a **Remyx API key** (from engine.remyx.ai → Settings) and an **Anthropic API key** (from console.anthropic.com). Set the Remyx key once:

```bash
export REMYXAI_API_KEY=...   # from engine.remyx.ai → Settings
```

#### Option A — `outrider init` (via the Remyx GitHub App) — recommended

```bash
# Set up on a repo, auto-creating a research interest from it:
$ remyxai outrider init --repo owner/name --auto-interest

# …or use an existing research interest (UUID from engine.remyx.ai):
$ remyxai outrider init --repo owner/name --interest <uuid>
```

If the Remyx App isn't installed on the repo yet, the command prints an install link and waits. Then the engine provisions the workflow, repo secrets, and a setup PR, and — in the default `auto` mode — merges it and starts the first run. Connect your Anthropic key once on the engine's Integrations page, or pass `--anthropic-key` (or set `ANTHROPIC_API_KEY`) and the CLI connects it for you.

`--mode`: `auto` (default — set it up and start the first run), `review` (open the setup PR for you to merge), `off` (just create the research interest).

#### Option B — `outrider setup-local` (no GitHub App)

For teams that can't install a third-party App yet. The CLI uses your own authenticated `gh` to do everything — nothing new to security-review.

```bash
$ export ANTHROPIC_API_KEY=...   # stored as a repo secret by the CLI

$ remyxai outrider setup-local --repo owner/name --auto-interest
```

Using your `gh` credentials, the CLI sets the `REMYX_API_KEY` + `ANTHROPIC_API_KEY` repo secrets, writes `.github/workflows/outrider.yml`, opens a setup PR, and — in `auto` mode — merges it and dispatches the first run. So the Action can open its recommendation PRs, the CLI enables the repo's *"Allow GitHub Actions to create and approve pull requests"* setting; the Action then uses the repo's built-in `GITHUB_TOKEN` (no GitHub token is stored as a secret — only `REMYX_API_KEY` and `ANTHROPIC_API_KEY`).

Requires `gh` authenticated (`gh auth login`, or `$GITHUB_TOKEN` with `repo` + `workflow` scopes) and admin on the target repo.

#### Common options

- `--repo owner/name` — target repo (or a GitHub URL); defaults to the current directory's git remote.
- `--interest <uuid>` / `--auto-interest` — use an existing research interest, or create one from the repo.
- `--mode auto|review` (`init` also has `off`) — how far to take setup.
- `--dry-run` — print the plan (and, for `setup-local`, the rendered workflow) and exit without making changes.
- `-y` / `--yes` — skip the confirmation prompt.

Preview either path safely before committing to it:

```bash
$ remyxai outrider setup-local --repo owner/name --auto-interest --dry-run
```

**A note on credentials:** with `setup-local`, your `REMYXAI_API_KEY` is stored as the repo's `REMYX_API_KEY` secret so the Action can fetch recommendations — anyone with write access to the repo's workflows can consume Remyx credits on that key, so use it on repos you control. With `outrider init`, the Remyx App provisions a scoped key for you. In both paths your Anthropic key is stored as a repo secret to run the agent.

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

