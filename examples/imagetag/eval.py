from app import LlamaChatInterface, DEFAULT_METADATA_PROMPT,DEFAULT_SEARCH_PROMPT

import base64
import json
import mimetypes
import os
import re
import shelve
import subprocess
from multiprocessing import freeze_support
from pathlib import Path
from typing import Generator, List, Optional

import gradio as gr
from dotenv import load_dotenv
import pandas as pd
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document



model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
#provider_name = "together"
provider_name = "meta-reference-inference"
os.environ["INFERENCE_MODEL"] = model_name
chat_interface = LlamaChatInterface()
chat_interface.initialize_system(provider_name)
folder_input = './example/'
#chat_interface.process_folder(folder_input)
#chat_interface.create_rag_from_imagestore()
ground_truth = pd.read_csv(folder_input + '/styles.csv', on_bad_lines='skip')
# def retrieval_test():
#     pass

def eval_tags(chat_interface,ground_truth):
    ground_truth = ground_truth.head(1000)
    # counter for each attribute's result
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
    result = {}
    output_file = './tags.json'
    for gold in ground_truth:
        idx = gold['id']
        image_path = folder_input + '/images/' + str(idx) + '.jpg'
        metadata = chat_interface.get_metadata(image_path)
        try:
            json_metadata = json.loads(metadata)
            result{idx} =  json_metadata
        except:
            print('tagging failed:', idx )
    print('Saving raw tags into: ', output_file)
    with open(output_file, 'a+') as f:
        f.write(json.dumps(result)+'\n')
    eval_result = {}
    total_item = len(results)
    for idx,json_metadata in results.items()
            for attribute in attributes_results.keys():
                if attribute in json_metadata:
                    # check if the generated value matches the ground truth value
                    eval_result[idx][attribute] = json_metadata[attribute] == gold[convert_dict[attribute]]
                    attributes_results[attribute] +=1
    eval_log_file = './eval_log.json'
    print('Saving eval logs into: ', eval_log_file)
    with open(eval_log_file, 'a+') as f:
        f.write(json.dumps(eval_result)+'\n')
    eval_output_file = './eval_output.json'
    for attribute in attributes_results.keys():
        acc =  attributes_results[attribute] / total_item
        print('attribute: ', attribute, ' has accuray: ', acc)
        attributes_results[attribute+'_acc'] = acc 
    with open(eval_output_file, 'a+') as f:
        f.write(json.dumps(attributes_results)+'\n')
    
eval_tags(chat_interface,ground_truth)
