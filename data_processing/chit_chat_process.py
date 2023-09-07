import pickle
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, pipeline, AutoTokenizer

from helper import get_emo_counts, get_js_distance, load_emo_classifier
import torch
from tqdm import tqdm

load_path_prefix = '../'
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
set = "validation"
seed = 42
size = 10000

# load chitchat dataset
dataset_path = f'{load_path_prefix}modeldata/chitchat_dataset.p'

with open(dataset_path, "rb") as f:
    [ed_ds] = pickle.load(f)
#dataset = Dataset.from_dict(ed_ds)
dataset = ed_ds.shuffle(seed=seed).select(range(size))
f.close()
"""
#dataset = load_dataset('Salesforce/dialogstudio', "Prosocial")
#dataset = load_dataset("medical_dialog", "en", verification_mode='no_checks')

f = open(f"../modeldata/medical_dialog_train_dataset.p", 'wb')# _dis_score
pickle.dump([dataset], f)
f.close()
#tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
"""

# load empathy classifier
emp_classifier_model = f"{load_path_prefix}models/roberta-empathy-03-06-2023-18_21_58"
empathy_model_id = emp_classifier_model
empathy_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
empathy_model = RobertaForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32).to(device)
# change
#empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True)
empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True, device=0)

# load emotion classifier
emo_classifier = load_emo_classifier(device)

# initialise
conv_id = ""
prev_context = ""
target = []
new_prompt = []
prompt_emo = []
target_emo = []
top_count = 10
threshold = 0.4

for d in tqdm(dataset[set]):
    for conv in d["log"]:
        prompt = conv["user utterance"]
        resp = conv["system response"]

        # test empathy level
        emp_results = empathy_classifier(resp, truncation=True, max_length=512)[0]
        label = emp_results['label']

        # skip data if classified as "No Empathy" represented by "LABEL_0"
        if label != "LABEL_0":
            prev_emo = emo_classifier(prompt)
            utt_emo = emo_classifier(resp)
            _, score = get_emo_counts(prev_emo, utt_emo, top=top_count)
            prev_emo = prev_emo[0][0]["label"]
            utt_emo = utt_emo[0][0]["label"]

            # skip data if emo_count score lower than threshold
            if score >= threshold:
                prompt_emo.append(prev_emo)
                target_emo.append(utt_emo)
                #prompt = d["prompt"]
                utterance = resp
                new_prompt.append(f"[{prev_emo}] {prompt}")
                target.append(f"[{utt_emo}] {utterance}")
            #prev_context = utterance

# create a dict for saving
dict = {"prompt" : new_prompt,
        "target" : target,
        "prompt_emo": prompt_emo,
        "target_emo": target_emo}

# save the resulting dataset
# local: '../', remote: ''
f = open(f"modeldata/sp_token_ws_empathy_clean_count_top{top_count}_score{threshold}_emo_{set}_chitchat_dataset.p", 'wb')# _dis_score
pickle.dump([dict], f)
f.close()
