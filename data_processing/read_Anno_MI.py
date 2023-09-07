import csv
import pickle

import torch
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, pipeline

from helper import load_emo_classifier, get_emo_counts

load_path_prefix = '../'
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

# load empathy classifier
emp_classifier_model = f"{load_path_prefix}models/roberta-empathy-03-06-2023-18_21_58"
empathy_model_id = emp_classifier_model
empathy_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
empathy_model = RobertaForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32).to(device)
# change
empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True)
#empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True, device=0)

# load emotion classifier
emo_classifier = load_emo_classifier(device)

# initialise
conv_id = ""
prev_context = ""
target = []
new_prompt = []
prompt_emo = []
target_emo = []
dia_context = []
top_count = 10
threshold = 0.4
count = 0
"""
"transcript_id": 1
"utterance_text": 8
"""
with open("../modeldata/AnnoMI-full.csv", 'r') as file:
    csvreader = csv.reader(file) #csvreader = csv.reader(file, delimiter="\t")
    for d in tqdm(csvreader):
        # skip the column headers
        if count == 0:
            count += 1
            continue

        # new conversation, reset
        if d[1] != conv_id:
            conv_id = d[1]
            prev_context = d[8]
            #context = d["prompt"]
        else:
            # test empathy level
            utterance = d[8]
            emp_results = empathy_classifier(d[8], padding='max_length', truncation=True, max_length=512)[0]
            label = emp_results['label']

            # skip data if classified as "No Empathy" represented by "LABEL_0"
            if label != "LABEL_0":
                prev_emo = emo_classifier(prev_context)
                utt_emo = emo_classifier(d[8])
                _, score = get_emo_counts(prev_emo, utt_emo, top=top_count)
                prev_emo = prev_emo[0][0]["label"]
                utt_emo = utt_emo[0][0]["label"]

                # skip data if emo_count score lower than threshold
                if score >= threshold:
                    prompt_emo.append(prev_emo)
                    target_emo.append(utt_emo)

                    # no emotion special token needed for test set
                    if set == "test":
                        new_prompt.append(f"{prev_context}")
                        target.append(f"{utterance}")
                        #dia_context.append(f"{context}")
                    else:
                        new_prompt.append(f"[{prev_emo}] {prev_context}")
                        target.append(f"[{utt_emo}] {utterance}")
                        #dia_context.append(f"{context}")
            prev_context = utterance

# create a dict for saving
dict = {"prompt" : new_prompt,
        "target" : target,
        "prompt_emo": prompt_emo,
        "target_emo": target_emo}

# save the resulting dataset
# local: '../', remote: ''
f = open(f"../modeldata/sp_token_ws_empathy_clean_count_top{top_count}_score{threshold}_emo_AnnoMI_dataset.p", 'wb')
pickle.dump([dict], f)
f.close()