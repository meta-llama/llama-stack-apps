import base64
import json
import mimetypes
import os
from multiprocessing import freeze_support
from openai import OpenAI

from pathlib import Path
from typing import Generator, List, Optional

import pandas as pd
import evaluate

from app import DEFAULT_METADATA_PROMPT, DEFAULT_SEARCH_PROMPT, LlamaChatInterface
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document


#provider_name = "together"
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

provider_name = "remote-vllm"
os.environ["INFERENCE_MODEL"] = model_id
os.environ["VLLM_URL"] = 'http://localhost:8000/v1'
chat_interface = LlamaChatInterface()
chat_interface.initialize_system(provider_name)
chat_interface.set_model_name(model_id)
folder_input = "/home/kaiwu/work/image_tags_data/fashion-dataset"
# chat_interface.process_folder(folder_input)
# chat_interface.create_rag_from_imagestore()
ground_truth = pd.read_csv(folder_input + "/styles.csv", on_bad_lines="skip")


# def retrieval_test():
#     pass



def generate_tags(chat_interface, ground_truth):
    ground_truth = ground_truth.head(1000)
    print("start tagging")
    # counter for each attribute's result
    results = {}
    output_file = "./tags.json"
    for _ , gold in ground_truth.iterrows():
        print(gold)
        idx = gold["id"]
        image_path = folder_input + "/images/" + str(idx) + ".jpg"
        metadata = chat_interface.get_metadata_from_image(image_path)
        try:
            json_metadata = json.loads(metadata)
            results[idx] = json_metadata
        except:
            print("tagging failed:", idx)
    print("Saving raw tags into: ", output_file)
    with open(output_file, "a+") as f:
        f.write(json.dumps(results) + "\n")

def eval_tags(output_file):
    print("start eval")
    with open(output_file, "r") as f:
        results = json.loads(f.read())
    eval_result = {}
    attributes_results = {
        "product_description": 0,
        "category": 0,
        "product_type": 0,
        "product_color": 0,
        "gender": 0,
    }
    # find correct attributes in ground truth using this convert_dict
    convert_dict = {
        "product_description": "productDisplayName",
        "category": "subCategory",
        "product_type": "articleType",
        "product_color": "baseColour",
        "gender": "gender",
    }
    total_item = len(results.keys())
    print("total item: ", total_item)
    for idx, json_metadata in results.items():
        idx = int(idx)
        eval_result[idx] = {}
        for attribute in attributes_results.keys():
            gold = ground_truth[ground_truth["id"] == idx].iloc[0]
            if attribute in json_metadata:
                # check if the generated value matches the ground truth value
                pred = json_metadata[attribute].strip().lower()
                gt = gold[convert_dict[attribute]].strip().lower()
                print("attribute: ", attribute, " pred: ", pred, " gt: ", gt) 
                eval_result[idx][attribute] = pred == gt
                attributes_results[attribute] += pred == gt
    eval_log_file = "./eval_log.json"
    print("Saving eval logs into: ", eval_log_file)
    with open(eval_log_file, "a+") as f:
        f.write(json.dumps(eval_result) + "\n")
    eval_output_file = "./eval_output.json"
    acc_result = {}
    for attribute in attributes_results.keys():
        acc = attributes_results[attribute] / total_item
        print("attribute: ", attribute, " has accuray: ", acc)
        acc_result[attribute + "_acc"] = acc
    with open(eval_output_file, "a+") as f:
        f.write(json.dumps(acc_result) + "\n")
        f.write(json.dumps(attributes_results) + "\n")

def compute_judge_score(questions: list, generated : list, reference: list, api_config,api_url="http://localhost:8001/v1",key="EMPTY"):
    correct_num = 0
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    llm = OpenAI(
        openai_api_key=key,
        openai_api_base=api_url,
        model_name=model_name,
        max_tokens=1000,
        temperature=0.0)
    all_tasks = []
    for question,prediction,gold in zip(questions, generated,reference):
        message = api_config['judge_prompt_template'].format(question=question,prediction=prediction,gold=gold)
        all_tasks.append(message)
    judge_responses = llm.batch(all_tasks)
    judge_responses = ["YES" in item.content for item in judge_responses]
    correct_num = sum(judge_responses)
    return correct_num/len(questions),judge_responses
def compute_rouge_score(generated : list, reference: list):
    rouge_score = evaluate.load('rouge')
    return rouge_score.compute(
        predictions=generated,
        references=reference,
        use_stemmer=True,
        use_aggregator=True
    )

if __name__ == "__main__":
    freeze_support()
    eval_tags(chat_interface, ground_truth)
