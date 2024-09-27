# llama-stack-apps

[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/TZAAYNVtrU)

This repo shows examples of applications built on top of [Llama Stack](https://github.com/meta-llama/llama-stack). Starting Llama 3.1 you can build agentic applications capable of:

- breaking a task down and performing multi-step reasoning.
- using tools to perform some actions
  - built-in: the model has built-in knowledge of tools like search or code interpreter
  - zero-shot: the model can learn to call tools using previously unseen, in-context tool definitions
- providing system level safety protections using models like Llama Guard.

> [!NOTE]
> The Llama Stack API is still evolving and may change. Feel free to build and experiment, but please don't rely on its stability just yet!


An agentic app requires a few components:
- ability to run inference on the underlying Llama series of models
- ability to run safety checks using the Llama Guard series of models
- ability to execute tools, including a code execution environment, and loop using the model's multi-step reasoning process

All of these components are now offered by a single Llama Stack Distribution. The [Llama Stack](https://github.com/meta-llama/llama-stack) defines and standardizes these components and many others that are needed to make building Generative AI applications smoother. Various implementations of these APIs are then assembled together via a **Llama Stack Distribution**.

# Getting started with the Llama Stack Distributions

To get started with Llama Stack Distributions, you'll need to:

1. Install prerequisites
2. Download the model checkpoints
3. Build and start a Llama Stack server
4. Connect your client agentic app to Llama Stack server

Once started, you can then just point your agentic app to the URL for this server (e.g. `http://localhost:5000`).

## 1. Install Prerequisites

**Python Packages**

We recommend creating an isolated conda Python environment.

```bash
# Create and activate a virtual environment
ENV=stack
conda create -n $ENV python=3.10
cd <path-to-llama-stack-apps-repo>
conda activate $ENV

# Install dependencies
pip install -r requirements.txt
```

This will install all dependencies required to (1) Build and start a Llama Stack server (2) Connect your client app to Llama Stack server.

**CLI Packages**


With `llama-stack` installed, you should be able to use the Llama Stack CLI and run `llama --help`. Please checkout our [CLI Reference](https://github.com/meta-llama/llama-stack/blob/main/docs/cli_reference.md) for more details.

```bash
usage: llama [-h] {download,model,stack} ...

Welcome to the Llama CLI

options:
  -h, --help            show this help message and exit

subcommands:
  {download,model,stack}
```

## Download Model Checkpoints

#### Downloading from [Meta](https://llama.meta.com/llama-downloads/)

Download the required checkpoints using the following commands:
```bash
# download the 8B model, this can be run on a single GPU
llama download --source meta --model-id Llama3.1-8B-Instruct --meta-url META_URL

# you can also get the 70B model, this will require 8 GPUs however
llama download --source meta --model-id Llama3.1-70B-Instruct --meta-url META_URL

# llama-agents have safety enabled by default. For this, you will need
# safety models -- Llama-Guard and Prompt-Guard
llama download --source meta --model-id Prompt-Guard-86M --meta-url META_URL
llama download --source meta --model-id Llama-Guard-3-8B --meta-url META_URL
```

For all the above, you will need to provide a URL (META_URL) which can be obtained from https://llama.meta.com/llama-downloads/ after signing an agreement.

#### Downloading from [Huggingface](https://huggingface.co/meta-llama)

Essentially, the same commands above work, just replace `--source meta` with `--source huggingface`.

```bash
llama download --source huggingface --model-id  Meta-Llama3.1-8B-Instruct --hf-token <HF_TOKEN>

llama download --source huggingface --model-id Meta-Llama3.1-70B-Instruct --hf-token <HF_TOKEN>

llama download --source huggingface --model-id Llama-Guard-3-8B --ignore-patterns *original*
llama download --source huggingface --model-id Prompt-Guard-86M --ignore-patterns *original*
```

**Important:** Set your environment variable `HF_TOKEN` or pass in `--hf-token` to the command to validate your access. You can find your token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

> **Tip:** Default for `llama download` is to run with `--ignore-patterns *.safetensors` since we use the `.pth` files in the `original` folder. For Llama Guard and Prompt Guard, however, we need safetensors. Hence, please run with `--ignore-patterns original` so that safetensors are downloaded and `.pth` files are ignored.

#### Downloading via Ollama

If you're already using ollama, we also have a supported Llama Stack distribution `local-ollama` and you can continue to use ollama for managing model downloads.

```
ollama pull llama3.1:8b-instruct-fp16
ollama pull llama3.1:70b-instruct-fp16
```

> [!NOTE]
> Only the above two models are currently supported by Ollama.


## Build, Configure, Run Llama Stack Distribution
- Please see our [Getting Started Guide](https://github.com/meta-llama/llama-stack/blob/main/docs/getting_started.md) for more details on setting up a Llama Stack distribution and running server to serve API endpoints.


### Step 1. Build
In the following steps, imagine we'll be working with a `Meta-Llama3.1-8B-Instruct` model. We will name our build `8b-instruct` to help us remember the config. We will start build our distribution (in the form of a Conda environment, or Docker image). In this step, we will specify:
- `name`: the name for our distribution (e.g. `8b-instruct`)
- `image_type`: our build image type (`conda | docker`)
- `distribution_spec`: our distribution specs for specifying API providers
  - `description`: a short description of the configurations for the distribution
  - `providers`: specifies the underlying implementation for serving each API endpoint
  - `image_type`: `conda` | `docker` to specify whether to build the distribution in the form of Docker image or Conda environment.

#### Build a local distribution with conda
The following command and specifications allows you to get started with building.
```
llama stack build
```
- You'll be prompted to enter build information interactively.
```
llama stack build

> Enter an unique name for identifying your Llama Stack build distribution (e.g. my-local-stack): 8b-instruct
> Enter the image type you want your distribution to be built with (docker or conda): conda

 Llama Stack is composed of several APIs working together. Let's configure the providers (implementations) you want to use for these APIs.
> Enter the API provider for the inference API: (default=meta-reference): meta-reference
> Enter the API provider for the safety API: (default=meta-reference): meta-reference
> Enter the API provider for the agents API: (default=meta-reference): meta-reference
> Enter the API provider for the memory API: (default=meta-reference): meta-reference
> Enter the API provider for the telemetry API: (default=meta-reference): meta-reference

 > (Optional) Enter a short description for your Llama Stack distribution:

Build spec configuration saved at ~/.conda/envs/llamastack-my-local-stack/8b-instruct-build.yaml
You can now run `llama stack configure my-local-stack`
```

#### (Alternative) Downloading Pre-built Docker image

We provide 2 pre-built Docker image of Llama Stack distribution, which can be found in the following links.
- [llamastack-local-gpu](https://hub.docker.com/repository/docker/llamastack/llamastack-local-gpu/general)
  - This is a packaged version with our local meta-reference implementations, where you will be running inference locally with downloaded Llama model checkpoints.
- [llamastack-local-cpu](https://hub.docker.com/repository/docker/llamastack/llamastack-local-cpu/general)
   - This is a lite version with remote inference where you can hook up to your favourite remote inference framework (e.g. ollama, fireworks, together, tgi) for running inference without GPU.

> [!NOTE]
> For GPU inference, you need to set these environment variables for specifying local directory containing your model checkpoints, and enable GPU inference to start running docker container.
```
export LLAMA_CHECKPOINT_DIR=~/.llama
```


To download and start running a pre-built docker container, you may use the following commands:
```
docker image pull llamastack/llamastack-local-gpu
llama stack configure llamastack-local-gpu
llama stack run local-gpu
```

### Step 2. Configure
After our distribution is built (either in form of docker or conda environment), we will run the following command to
```
llama stack configure [<name> | <path/to/name.build.yaml> | <docker-image-name>]
```
- For `conda` environments: <path/to/name.build.yaml> would be the generated build spec saved from Step 1.
- For `docker` images downloaded from Dockerhub, you could also use <docker-image-name> as the argument.
   - Run `docker images` to check list of available images on your machine.

```
$ llama stack configure 8b-instruct

Configuring API: inference (meta-reference)
Enter value for model (existing: Meta-Llama3.1-8B-Instruct) (required):
Enter value for quantization (optional):
Enter value for torch_seed (optional):
Enter value for max_seq_len (existing: 4096) (required):
Enter value for max_batch_size (existing: 1) (required):

Configuring API: memory (meta-reference-faiss)

Configuring API: safety (meta-reference)
Do you want to configure llama_guard_shield? (y/n): y
Entering sub-configuration for llama_guard_shield:
Enter value for model (default: Llama-Guard-3-8B) (required):
Enter value for excluded_categories (default: []) (required):
Enter value for disable_input_check (default: False) (required):
Enter value for disable_output_check (default: False) (required):
Do you want to configure prompt_guard_shield? (y/n): y
Entering sub-configuration for prompt_guard_shield:
Enter value for model (default: Prompt-Guard-86M) (required):

Configuring API: agentic_system (meta-reference)
Enter value for brave_search_api_key (optional):
Enter value for bing_search_api_key (optional):
Enter value for wolfram_api_key (optional):

Configuring API: telemetry (console)

YAML configuration has been written to ~/.llama/builds/conda/8b-instruct-run.yaml
```

After this step is successful, you should be able to find a run configuration spec in `~/.llama/builds/conda/8b-instruct-run.yaml` with the following contents. You may edit this file to change the settings.

As you can see, we did basic configuration above and configured:
- inference to run on model `Meta-Llama3.1-8B-Instruct` (obtained from `llama model list`)
- Llama Guard safety shield with model `Llama-Guard-3-8B`
- Prompt Guard safety shield with model `Prompt-Guard-86M`

For how these configurations are stored as yaml, checkout the file printed at the end of the configuration.

Note that all configurations as well as models are stored in `~/.llama`


### Step 3. Run
Now, let's start the Llama Stack Distribution Server. You will need the YAML configuration file which was written out at the end by the `llama stack configure` step.

```
llama stack run 8b-instruct
```

You should see the Llama Stack server start and print the APIs that it is supporting

```
$ llama stack run 8b-instruct

> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Loaded in 19.28 seconds
NCCL version 2.20.5+cuda12.4
Finished model load YES READY
Serving POST /inference/batch_chat_completion
Serving POST /inference/batch_completion
Serving POST /inference/chat_completion
Serving POST /inference/completion
Serving POST /safety/run_shields
Serving POST /agents/memory_bank/attach
Serving POST /agents/create
Serving POST /agents/session/create
Serving POST /agents/turn/create
Serving POST /agents/delete
Serving POST /agents/session/delete
Serving POST /agents/memory_bank/detach
Serving POST /agents/session/get
Serving POST /agents/step/get
Serving POST /agents/turn/get
Listening on :::5000
INFO:     Started server process [453333]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
```

> [!NOTE]
> Configuration is in `~/.llama/builds/local/conda/8b-instruct.yaml`. Feel free to increase `max_seq_len`.

> [!IMPORTANT]
> The "local" distribution inference server currently only supports CUDA. It will not work on Apple Silicon machines.

> [!TIP]
> You might need to use the flag `--disable-ipv6` to  Disable IPv6 support

This server is running a Llama model locally.

## Test agents demo script

We have built sample demo scripts for interating with the Stack server.

With the server running, you may run to test out an simple Agent
```
python -m examples.agents.hello localhost 5000
```

You will see outputs of the form --
```
> created agents with agent_id=d050201b-0ca1-4abd-8eee-3cba2b8c0fbc
User> Hello
shield_call> No Violation
inference> How can I assist you today?
shield_call> No Violation
User> Which players played in the winning team of the NBA western conference semifinals of 2024, please use tools
shield_call> No Violation
inference> brave_search.call(query="NBA Western Conference Semifinals 2024 winning team players")
tool_execution> Tool:brave_search Args:{'query': 'NBA Western Conference Semifinals 2024 winning team players'}
tool_execution> Tool:brave_search Response:{"query": "NBA Western Conference Semifinals 2024 winning team players", "top_k": [{"title": "2024 NBA Western Conference Semifinals - Mavericks vs. Thunder | Basketball-Reference.com", "url": "https://www.basketball-reference.com/playoffs/2024-nba-western-conference-semifinals-mavericks-vs-thunder.html", "description": "Summary and statistics for the <strong>2024</strong> <strong>NBA</strong> <strong>Western</strong> <strong>Conference</strong> <strong>Semifinals</strong> - Mavericks vs. Thunder", "type": "search_result"}, {"title": "2024 NBA playoffs - Wikipedia", "url": "https://en.wikipedia.org/wiki/2024_NBA_playoffs", "description": "Aged 20 years and 96 days old, ... youngest <strong>player</strong> <strong>in</strong> <strong>NBA</strong> history to record 10+ points and 15+ rebounds in a playoff game, coming during game 6 of the Maverick&#x27;s <strong>Western</strong> <strong>Conference</strong> <strong>Semifinal</strong> <strong>win</strong> against the Thunder on May 18. The Timberwolves overcame a 20\u2013point deficit to <strong>win</strong> game 7 against the Nuggets, the largest game 7 comeback in <strong>NBA</strong> playoffs history. With the defending champion Nuggets losing to the Minnesota Timberwolves, the <strong>2024</strong> playoffs marked ...", "type": "search_result"}, {"title": "2024 NBA Playoffs | Official Bracket, Schedule and Series Matchups", "url": "https://www.nba.com/playoffs/2024", "description": "The official site of the <strong>2024</strong> <strong>NBA</strong> Playoffs. Latest news, schedules, matchups, highlights, bracket and more.", "type": "search_result"}]}
shield_call> No Violation
inference> The players who played in the winning team of the NBA Western Conference Semifinals of 2024 are not specified in the search results provided. However, the search results suggest that the Mavericks played against the Thunder in the Western Conference Semifinals, and the Mavericks won the series.
shield_call> No Violation
```

## Start an App and Interact with the Server

Now that the Stack server is setup, the next thing would be to run an agentic app using Agents APIs.

We have built sample scripts, notebooks and a UI chat interface ( using [Mesop]([url](https://google.github.io/mesop/)) ! ) to help you get started.

Start an app (local) and interact with it by running the following command:
```bash
PYTHONPATH=. mesop app/main.py
```
This will start a mesop app and you can go to `localhost:32123` to play with the chat interface.

<img src="demo.png" alt="Chat App" width="600"/>

Optionally, you can setup API keys for custom tools:
- [WolframAlpha](https://developer.wolframalpha.com/): store in `WOLFRAM_ALPHA_API_KEY` environment variable
- [Brave Search](https://brave.com/search/api/): store in `BRAVE_SEARCH_API_KEY` environment variable

Similar to this main app, you can also try other variants
- `PYTHONPATH=. mesop app/chat_with_custom_tools.py`  to showcase how custom tools are integrated
- `PYTHONPATH=. mesop app/chat_moderation_with_llama_guard.py`  to showcase how the app is modified to act as a chat moderator for safety

## Create agentic systems and interact with the Stack server

NOTE: Ensure that Stack server is still running.

```bash
cd <path-to-llama-agentic-system>
conda activate $ENV
llama stack run <name> # If not already started

PYTHONPATH=. python examples/scripts/vacation.py localhost 5000
```

You should see outputs to stdout of the form --
```bash
Environment: ipython
Tools: brave_search, wolfram_alpha, photogen

Cutting Knowledge Date: December 2023
Today Date: 23 July 2024


User> I am planning a trip to Switzerland, what are the top 3 places to visit?
Final Llama Guard response shield_type=<BuiltinShield.llama_guard: 'llama_guard'> is_violation=False violation_type=None violation_return_message=None
Ran PromptGuardShield and got Scores: Embedded: 0.9999765157699585, Malicious: 1.1110752893728204e-05
StepType.shield_call> No Violation
role='user' content='I am planning a trip to Switzerland, what are the top 3 places to visit?'
StepType.inference> Switzerland is a beautiful country with a rich history, culture, and natural beauty. Here are three must-visit places to add to your itinerary: ....

```

> **Tip** You can optionally do `--disable-safety` in the scripts to avoid running safety shields all the time.


Feel free to reach out if you have questions.


## Develop in your preferred language
- Check out our client SDKs for connecting to Llama Stack server, you can choose from [python](https://github.com/meta-llama/llama-stack-client-python), [node](https://github.com/meta-llama/llama-stack-client-node), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) programming languages to quickly build your applications.


## Using VirtualEnv instead of Conda

> [!NOTE]
> While you can run the apps using `venv`, installation of a distribution requires conda.

#### In Linux

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
```

#### For Windows

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # For Command Prompt
# or
.\venv\Scripts\Activate.ps1  # For PowerShell
# or
source venv\Scripts\activate  # For Git
```

The instructions thereafter (including `pip install -r requirements.txt` for installing the dependencies) remain the same.
