import pickle
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, pipeline, AutoTokenizer

from helper import get_emo_counts, get_js_distance, load_emo_classifier
import torch

load_path_prefix = ''
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
dataset = load_dataset("empathetic_dialogues")
#dataset = load_dataset("allenai/real-toxicity-prompts")
set = "train"


#tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")


"""
prompt = dataset[set]["prompt"]
utterance = dataset[set]["continuation"]
ds_prompt = Dataset.from_list(prompt)["text"]
ds_utterance = Dataset.from_list(utterance)["text"]

dict = {"prompt" : ds_prompt,
        "target" : ds_utterance}

f = open("../modeldata/train_real_toxicity_dataset.p", 'wb')
pickle.dump([dict], f)
f.close()
"""

emp_classifier_model = f"{load_path_prefix}models/roberta-empathy-03-06-2023-18_21_58"
empathy_model_id = emp_classifier_model
empathy_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
empathy_model = RobertaForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32).to(device)
# change
#empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True)
empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True, device=0)



emo_classifier = load_emo_classifier(device)
conv_id = ""
prev_context = ""
target = []
new_prompt = []
prompt_emo = []
target_emo = []
top_count = 10
threshold = 0.4

for d in dataset[set]:
    if d["conv_id"] != conv_id:
        conv_id = d["conv_id"]
        prev_context = d["utterance"]
    else:
        # test
        emp_results = empathy_classifier(d["utterance"], padding='max_length', truncation=True, max_length=512)[0]
        label = emp_results['label']
        if label != "LABEL_0":
            prev_emo = emo_classifier(prev_context)
            utt_emo = emo_classifier(d["utterance"])
            _, score = get_emo_counts(prev_emo, utt_emo, top=top_count)
            prev_emo = prev_emo[0][0]["label"]
            utt_emo = utt_emo[0][0]["label"]
            if score >= threshold:
                prompt_emo.append(prev_emo)
                target_emo.append(utt_emo)
                #prev_context = f"Input Emotion: {prev_emo}, Input: {prev_context}, Response Emotion: {utt_emo}"
                new_prompt.append(f"[{prev_emo}] {prev_context}")
                utterance = d["utterance"]
                target.append(f"[{utt_emo}] {utterance}")
                prev_context = utterance

print(len(new_prompt))
dict = {"prompt" : new_prompt,
        "target" : target,
        "prompt_emo": prompt_emo,
        "target_emo": target_emo}

#dataset = Dataset.from_dict(dict)
# local: '../', remote: ''
f = open(f"modeldata/sp_token_ws_empathy_clean_count_top{top_count}_score{threshold}_emo_{set}_dialogue_dataset.p", 'wb')# _dis_score
pickle.dump([dict], f)
f.close()


"""
# checks
print(new_prompt[:7])
print(target[:7])
print(len(new_prompt) == len(target))
"""
