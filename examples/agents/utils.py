from llama_stack_client import LlamaStackClient
from termcolor import colored


def check_model_is_available(client: LlamaStackClient, model: str):
    available_models = [
        model.identifier
        for model in client.models.list()
        if model.model_type == "llm" and "guard" not in model.identifier
    ]

    if model not in available_models:
        print(
            colored(
                f"Model `{model}` not found. Available models:\n\n{available_models}\n",
                "red",
            )
        )
        return False

    return True


def get_any_available_model(client: LlamaStackClient):
    available_models = [
        model.identifier
        for model in client.models.list()
        if model.model_type == "llm" and "guard" not in model.identifier
    ]
    if not available_models:
        print(colored("No available models.", "red"))
        return None

    return available_models[0]
