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

**Getting started with the Llama Stack Distributions**
========================================

To get started with Llama Stack Distributions, you'll need to:

1. Install prerequisites
2. Setup the toolchain which provides the core `llama` CLI
3. Download the models
4. Build a Llama Stack Distribution image
5. Start the Llama Stack server

Once started, you can then just point your agentic app to the URL for this server (e.g. `http://localhost:5000`).


**1. Install Prerequisites**
-----------------------------
**Python Packages**

We recommend creating an isolated conda Python environment.

```bash
# Create and activate a virtual environment
ENV=app_env
conda create -n $ENV python=3.10
cd <path-to-llama-stack-apps-repo>
conda activate $ENV

# Install dependencies
pip install -r requirements.txt
```

You should now be able to run `llama --help`:

```bash
usage: llama [-h] {download,model,api,stack} ...

Welcome to the LLama cli

options:
  -h, --help            show this help message and exit

subcommands:
  {download,model,api,stack}
```

**bubblewrap**

The code execution environment uses [bubblewrap](https://github.com/containers/bubblewrap) for isolation. This may already be installed on your system; if not, it's likely in your OS's package repository.

**Ollama (optional)**

If you plan to use Ollama for inference, you'll need to install the server [via these instructions](https://ollama.com/download).


**Download Checkpoints**
----------------------------------------------

#### Downloading from [Meta](https://llama.meta.com/llama-downloads/)

Download the required checkpoints using the following commands:
```bash
# download the 8B model, this can be run on a single GPU
llama download --source meta --model-id Meta-Llama3.1-8B-Instruct --meta-url META_URL

# you can also get the 70B model, this will require 8 GPUs however
llama download --source meta --model-id Meta-Llama3.1-70B-Instruct --meta-url META_URL

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


**Installing and Configuring Distributions**
------------------------------------

> [!NOTE]
> `local` distribution has only been tested on linux as of now.
> For other platforms (ubuntu, mac) try using the `local-ollama` distribution and install platform specific ollama.


Let’s start with listing available distributions
```
$ llama stack list-distributions

+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| Distribution ID                | Providers                             | Description                                                          |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| local                          | {                                     | Use code from `llama_toolchain` itself to serve all llama stack APIs |
|                                |   "inference": "meta-reference",      |                                                                      |
|                                |   "memory": "meta-reference-faiss",   |                                                                      |
|                                |   "safety": "meta-reference",         |                                                                      |
|                                |   "agentic_system": "meta-reference"  |                                                                      |
|                                | }                                     |                                                                      |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| remote                         | {                                     | Point to remote services for all llama stack APIs                    |
|                                |   "inference": "remote",              |                                                                      |
|                                |   "safety": "remote",                 |                                                                      |
|                                |   "agentic_system": "remote",         |                                                                      |
|                                |   "memory": "remote"                  |                                                                      |
|                                | }                                     |                                                                      |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| local-ollama                   | {                                     | Like local, but use ollama for running LLM inference                 |
|                                |   "inference": "remote::ollama",      |                                                                      |
|                                |   "safety": "meta-reference",         |                                                                      |
|                                |   "agentic_system": "meta-reference", |                                                                      |
|                                |   "memory": "meta-reference-faiss"    |                                                                      |
|                                | }                                     |                                                                      |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| local-plus-fireworks-inference | {                                     | Use Fireworks.ai for running LLM inference                           |
|                                |   "inference": "remote::fireworks",   |                                                                      |
|                                |   "safety": "meta-reference",         |                                                                      |
|                                |   "agentic_system": "meta-reference", |                                                                      |
|                                |   "memory": "meta-reference-faiss"    |                                                                      |
|                                | }                                     |                                                                      |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
| local-plus-together-inference  | {                                     | Use Together.ai for running LLM inference                            |
|                                |   "inference": "remote::together",    |                                                                      |
|                                |   "safety": "meta-reference",         |                                                                      |
|                                |   "agentic_system": "meta-reference", |                                                                      |
|                                |   "memory": "meta-reference-faiss"    |                                                                      |
|                                | }                                     |                                                                      |
+--------------------------------+---------------------------------------+----------------------------------------------------------------------+
```

As you can see above, each “distribution” details the “providers” it is composed of. For example, `local` uses the “meta-reference” provider for inference while local-ollama relies on a different provider (Ollama) for inference. Similarly, you can use Fireworks or Together.AI for running inference as well.

To install a distribution, we run a simple command providing 2 inputs:
- **Distribution Id** of the distribution that we want to install ( as obtained from the list-distributions command )
- A **Name** for the specific build and configuration of this distribution.

Let's imagine you are working with a 8B-Instruct model. The following command will build a package (in the form of a Conda environment) _and_ configure it. As part of the configuration, you will be asked for some inputs (model_id, max_seq_len, etc.) We will name our build `8b-instruct` to help remember the config.

```
llama stack build local --name 8b-instruct
```

Once it runs successfully , you should see some outputs in the form:

```
$ llama stack build local --name 8b-instruct
....
....
Successfully installed cfgv-3.4.0 distlib-0.3.8 identify-2.6.0 libcst-1.4.0 llama_toolchain-0.0.2 moreorless-0.4.0 nodeenv-1.9.1 pre-commit-3.8.0 stdlibs-2024.5.15 toml-0.10.2 tomlkit-0.13.0 trailrunner-1.4.0 ufmt-2.7.0 usort-1.0.8 virtualenv-20.26.3

Successfully setup conda environment. Configuring build...

...
...

YAML configuration has been written to ~/.llama/builds/local/conda/8b-instruct.yaml
```

You can re-configure this distribution by running:
```
llama stack configure local --name 8b-instruct
```

Here is an example run of how the CLI will guide you to fill the configuration
```
$ llama stack configure local --name 8b-instruct

Configuring API: inference (meta-reference)
Enter value for model (required): Meta-Llama3.1-8B-Instruct
Enter value for quantization (optional):
Enter value for torch_seed (optional):
Enter value for max_seq_len (required): 4096
Enter value for max_batch_size (default: 1): 1
Configuring API: safety (meta-reference)
Do you want to configure llama_guard_shield? (y/n): y
Entering sub-configuration for llama_guard_shield:
Enter value for model (required): Llama-Guard-3-8B
Enter value for excluded_categories (required): []
Enter value for disable_input_check (default: False):
Enter value for disable_output_check (default: False):
Do you want to configure prompt_guard_shield? (y/n): y
Entering sub-configuration for prompt_guard_shield:
Enter value for model (required): Prompt-Guard-86M
...
...
YAML configuration has been written to ~/.llama/builds/local/conda/8b-instruct.yaml
```

As you can see, we did basic configuration above and configured:
- inference to run on model `Meta-Llama3.1-8B-Instruct` (obtained from `llama model list`)
- Llama Guard safety shield with model `Llama-Guard-3-8B`
- Prompt Guard safety shield with model `Prompt-Guard-86M`

For how these configurations are stored as yaml, checkout the file printed at the end of the configuration.

Note that all configurations as well as models are stored in `~/.llama`

**Installing and Configuring `local-ollama` Distribution**
----------------------------------------------

On one terminal, start ollama server using
```
ollama serve
```

> [!NOTE]
> In the server logs, you should see messages of the form `msg="llama runner started in xx seconds"` suggesting that the models are ready for inference.

You can test your ollama setup via
```
ollama run llama3.1:8b-instruct-fp16
```

Now, install the llama stack distribution:
```
llama stack build local-ollama --name 8b-instruct
```

**Starting up the Stack**
-------------------------

Now let’s start Llama Stack server.

You need the YAML configuration file which was written out at the end by the `llama stack build` step.


```
llama stack run local-ollama --name 8b-instruct --port 5000
```
You should see the Stack server start and print the APIs that it is supporting,

```
$ llama stack run local-ollama --name 8b-instruct --port 5000

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
Serving POST /agentic_system/memory_bank/attach
Serving POST /agentic_system/create
Serving POST /agentic_system/session/create
Serving POST /agentic_system/turn/create
Serving POST /agentic_system/delete
Serving POST /agentic_system/session/delete
Serving POST /agentic_system/memory_bank/detach
Serving POST /agentic_system/session/get
Serving POST /agentic_system/step/get
Serving POST /agentic_system/turn/get
Listening on :::5000
INFO:     Started server process [453333]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:5000 (Press CTRL+C to quit)
```


> [!NOTE]
> Configuration is in `~/.llama/builds/local-ollama/conda/8b-instruct.yaml`. Feel free to increase `max_seq_len`.

> [!IMPORTANT]
> The "local" distribution inference server currently only supports CUDA. It will not work on Apple Silicon machines.

This server is running a Llama model locally.

> [!TIP]
> You might need to use the flag `--disable-ipv6` to  Disable IPv6 support

Now that the Stack server is setup, the next thing would be to run an agentic app using AgenticSystem APIs.

We have built sample scripts, notebooks and a UI chat interface ( using [Mesop]([url](https://google.github.io/mesop/)) ! ) to help you get started.


**Add API Keys for Tools**
---------------------------------------------

API key configuration for the Agentic System will be asked by the `llama stack build` script when you install a Llama Stack distribution.

Tools that the model supports and which need API Keys --
- Brave for web search (https://api.search.brave.com/register)
- Wolfram for math operations (https://developer.wolframalpha.com/)

> **Tip** If you do not have API keys, you can still run the app without model having access to the tools.


**Start an App and Interact with the Server**
---------------------------------------------

Start an app (local) and interact with it by running the following command:
```bash
mesop app/main.py
```
This will start a mesop app and you can go to `localhost:32123` to play with the chat interface.

<img src="demo.png" alt="Chat App" width="600"/>

Optionally, you can setup API keys for custom tools:
- [WolframAlpha](https://developer.wolframalpha.com/): store in `WOLFRAM_ALPHA_API_KEY` environment variable
- [Brave Search](https://brave.com/search/api/): store in `BRAVE_SEARCH_API_KEY` environment variable

Similar to this main app, you can also try other variants
- `PYTHONPATH=. mesop app/chat_with_custom_tools.py`  to showcase how custom tools are integrated
- `PYTHONPATH=. mesop app/chat_moderation_with_llama_guard.py`  to showcase how the app is modified to act as a chat moderator for safety

**Create agentic systems and interact with the Stack server**
---------------------------------------------

NOTE: Ensure that Stack server is still running.

```bash
cd <path-to-llama-agentic-system>
conda activate $ENV
llama stack run local-ollama --name 8b --port 5000 # If not already started

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


**Using VirtualEnv instead of Conda**
-----------------------------
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
