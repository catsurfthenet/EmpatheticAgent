import pickle
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, pipeline, AutoTokenizer

from helper import get_emo_counts, get_js_distance, load_emo_classifier
import torch

load_path_prefix = '../'
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
#dataset = load_dataset("empathetic_dialogues")
#dataset = load_dataset("allenai/real-toxicity-prompts")
set = "test"

# load dataset
dataset_path = f'{load_path_prefix}modeldata/dialogue_dataset.p'#chitchat_dataset.p'

with open(dataset_path, "rb") as f:
    [ed_ds] = pickle.load(f)
dataset = Dataset.from_dict(ed_ds)
f.close()
"""
dataset = load_dataset('Salesforce/dialogstudio', "chitchat-dataset")
f = open(f"../modeldata/chitchat_dataset.p", 'wb')# _dis_score
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
conv_id = ""
prev_context = ""
target = []
new_prompt = []
prompt_emo = []
target_emo = []
top_count = 10
threshold = 0.4

for d in dataset: #[set]
    # test empathy level
    emp_results = empathy_classifier(d["target"], truncation=True, max_length=512)[0]
    label = emp_results['label']

    # skip data if classified as "No Empathy" represented by "LABEL_0"
    if label != "LABEL_0":
        prev_emo = emo_classifier(d["prompt"])
        utt_emo = emo_classifier(d["target"])
        _, score = get_emo_counts(prev_emo, utt_emo, top=top_count)
        prev_emo = prev_emo[0][0]["label"]
        utt_emo = utt_emo[0][0]["label"]

        # skip data if emo_count score lower than threshold
        if score >= threshold:
            prompt_emo.append(prev_emo)
            target_emo.append(utt_emo)
            prev_context = d["prompt"]
            utterance = d["target"]

            # no emotion special token needed for test set
            if set == "test":
                new_prompt.append(f"{prev_context}")
                target.append(f"{utterance}")
            else:
                new_prompt.append(f"[{prev_emo}] {prev_context}")
                target.append(f"[{utt_emo}] {utterance}")
            #prev_context = utterance

# create a dict for saving
dict = {"prompt" : new_prompt,
        "target" : target,
        "prompt_emo": prompt_emo,
        "target_emo": target_emo}

# save the resulting dataset
# local: '../', remote: ''
f = open(f"modeldata/ws_empathy_clean_count_top{top_count}_score{threshold}_emo_{set}_ED_dataset.p", 'wb')# _dis_score
pickle.dump([dict], f)
f.close()
