import os
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM, \
    RobertaTokenizerFast, RobertaForSequenceClassification, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset
import torch
from torch.utils.data import DataLoader
import random

# this code generates the text file containing prompts and responses generated from
# the 2 models (FT model and PLM) used in the questionnaire

# define which model to evaluate
seed = 1817
random.seed(seed)
load_path_prefix = ""
finetune = True
ft_model = "w1-1-1-1-0_50epochs-4250-loss1.32314"
pretrained_model = f"{load_path_prefix}models/local-facebook-blenderbot-400M-distill"
text_generation_model = "ppo_model" # blenderbot, ppo_model, dialogpt, opt_iml

# set path prefix
output_path_prefix = f"Questionnaire_Response_Generation{seed}_{ft_model}_blenderbot"
load_path_prefix = '' # change

device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
negative_sample_size = 8
#device = "mps"


# get trained model to be evaluated
if text_generation_model == "opt_iml":
    pretrained_model = "facebook/opt-iml-1.3b"
    model_id = pretrained_model
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)
elif text_generation_model == "dialogpt":
    pretrained_model = "microsoft/DialoGPT-small"
    model_id = pretrained_model
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)
elif text_generation_model == "ppo_model":
    model_id = ft_model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)
else:
    pretrained_model = "facebook/blenderbot-400M-distill"
    model_id = pretrained_model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)

blenderbot_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model, device_map={"": device}, torch_dtype=torch.float32).to(device)

if finetune or text_generation_model == "blenderbot":
    tokenizer_id = f"facebook/blenderbot-400M-distill"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    emo_labels = ['sadness', 'disappointment', 'neutral', 'fear', 'nervousness', 'disapproval', 'realization', 'annoyance', 'grief', 'approval', 'caring', 'remorse', 'disgust', 'desire', 'love', 'anger', 'embarrassment', 'joy', 'admiration', 'relief', 'surprise', 'optimism', 'confusion', 'curiosity', 'amusement', 'excitement', 'gratitude', 'pride']
    emo_labels = [f"[{i}]" for i in emo_labels]
    special_tokens_dict = {'additional_special_tokens': emo_labels}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    for i in emo_labels:
        tok_id = tokenizer.convert_tokens_to_ids(i)
    model.resize_token_embeddings(len(tokenizer))
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

#tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# load test dataset
with open(f"{load_path_prefix}modeldata/sp_token_context_ws_empathy_clean_count_top10_score0.4_emo_test_ED_dataset.p", "rb") as f: #ws_empathy_clean_count_top10_score0.4_emo_test_ED_dataset.p
    [test_dataset] = pickle.load(f)
#test_dataset = Dataset.from_dict(test_dataset)[:10]
test_dataset = Dataset.from_dict(test_dataset).shuffle(seed=seed).select(range(10))
#test_dataset = Dataset.from_dict(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# preprocessing
def tokenize(sample):
    prompt = sample["prompt"] # prompt
    continuation = sample["target"] # utterance

    sample["input_ids"] = tokenizer.encode(prompt, padding='max_length', truncation=True, max_length=128)
    sample["query"] = {"prompt": tokenizer.decode(sample["input_ids"]), "target": continuation}
    return sample

test_dataset = test_dataset.map(tokenize, batched=False)
test_dataset.set_format(type="torch")

# initialise text list
list_generated_texts = []
list_plm_generated_texts = []

with open(f'{output_path_prefix}_text_log.txt', 'w') as text_log:
    if text_generation_model == "ppo_model":
        text_log.write(f"FT Model name: {ft_model} \n")
    counter = 0
    prompts = []
    targets_list = []
    score_list = []
    for test_batch in enumerate(test_dataloader):
        test_query = test_batch[1]
        context = test_query["context"]
        context = [c.replace("_comma_", ",") for c in context]
        input_texts = test_query["prompt"]
        prompts = prompts + [it.replace("</s>", "").replace("_comma_", ",") for it in input_texts]
        #query_list.append(test_query["query"])
        target = test_query["target"]
        targets_list = targets_list + [t.replace("</s>", "").replace("_comma_", ",") for t in target]
        input_ids = tokenizer(input_texts, return_tensors="pt", padding='max_length', max_length=128, truncation=True).to(device)
        input_ids = input_ids["input_ids"]
        outputs = model.generate(input_ids, do_sample=False, num_beams=10, max_new_tokens=40, use_cache=True)
        plm_outputs = blenderbot_model.generate(input_ids, do_sample=False, num_beams=10, max_new_tokens=40, use_cache=True)
        #outputs = model.generate(**input_ids, num_beams=5, do_sample=True, max_new_tokens=40, use_cache=True, num_return_sequences=1)

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = [g.replace("<pad>", "").replace("_comma_", ",") for g in generated_texts]
        list_generated_texts = list_generated_texts + generated_texts

        plm_gen_texts = tokenizer.batch_decode(plm_outputs, skip_special_tokens=True)
        plm_gen_texts = [g.replace("<pad>", "").replace("_comma_", ",") for g in plm_gen_texts]
        list_plm_generated_texts = list_plm_generated_texts + plm_gen_texts

        input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        input_texts = [it.replace("_comma_", ",") for it in input_texts]

        random_no = []
        for t in range(len(input_texts)):
            text_log.write(f"Context: {context[t]} \n\n")
            print(f"Context: {context[t]}")
            text_log.write(f"Prompt: {input_texts[t]} \n\n")
            print(f"{counter}, {t} Prompt: {input_texts[t]}")

            r = random.random()
            if r < 0.5:
                text_log.write(f"Response 1: {generated_texts[t]} \n\n")
                print(f"{counter}, {t} Response 1: {generated_texts[t]}")
                text_log.write(f"Response 2: {plm_gen_texts[t]} \n\n\n")
                print(f"{counter}, {t} Response 2: {plm_gen_texts[t]}\n\n")
                random_no.append(0)
            else:
                text_log.write(f"Response 1: {plm_gen_texts[t]} \n\n")
                print(f"{counter}, {t} Response 1: {plm_gen_texts[t]}")
                text_log.write(f"Response 2: {generated_texts[t]} \n\n\n")
                print(f"{counter}, {t} Response 2: {generated_texts[t]}\n\n")
                random_no.append(1)
        counter += 1

        text_log.write(f"Random order (0-FT, 1-PLM): {random_no}\n\n")
text_log.close()

