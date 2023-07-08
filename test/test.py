import argparse
import csv
import os
import pickle
import evaluate
import numpy as np
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, \
    pipeline, RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import TextClassificationPipeline
#from classifiers import get_sentence_score
import nltk
from nltk.tokenize import word_tokenize
from helper import load_toxicity_classifier, load_empathy_classifier

load_path_prefix = '../'
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

context_length = 200

pretrained_model = "facebook/blenderbot-400M-distill"
ppo_model = "../models/DEV-blenderbot-400m-emo-probi-bleu-last-bleu0.06406307965517044" #"./gpt-neo-125M-emo-test5epoch"
#detox_model = "./gpt-neo-125M-detoxified-long-context-26-shl-1e4-final"
model_id = ppo_model
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
input_texts = ["The heat is driving me crazy. ", "I cannot swim."]
#inputs.input_ids = inputs.input_ids[:context_length]
#inputs.attention_mask = inputs.attention_mask[:context_length]

input_ids = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
outputs = model.generate(**input_ids, do_sample=True, max_new_tokens=40, use_cache=True)
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(generated_texts)

empathy_classifier = load_empathy_classifier(path_prefix="../")
result = empathy_classifier(generated_texts)
print(result)


toxicity_classifier = load_toxicity_classifier()
toxicity_result = toxicity_classifier(generated_texts)

print(empathy_classifier("I don't care, get away from me"))
print(toxicity_classifier("I don't care, get away from me"))

#inputs = tokenizer("ABC is a startup based in New York City and Paris", return_tensors = "pt")
ph = model(input_ids = outputs, labels = outputs)
loss = ph.loss
ppl = torch.exp(loss)
print(ppl)

"""
inputs_wiki_text = tokenizer("Generative Pretrained Transformer is an opensource artificial intelligence created by OpenAI in February 2019", return_tensors = "pt")
loss = model(input_ids = inputs_wiki_text["input_ids"], labels = inputs_wiki_text["input_ids"]).loss
ppl = torch.exp(loss)
"""


"""
if (os.path.exists('modeldata/concat_dataset.p')):
    print("LOADING concatenated empathetic_dialogue")
    with open('modeldata/concat_dataset.p', "rb") as f:
        [data] = pickle.load(f)
ds = Dataset.from_dict(data)

#for t in generated_texts:
#    score = get_sentence_score(t, ds)
#    print(score)
"""

"""
Possible errors:
1. problem with save_pretrained (HTTPError)
2. reward method
3. hyperparameters
4. PPO

Possible improvements:
1. use different datasets (1 + 2 ~ 1 week)
2. use 1 different model for comparison
3. error analysis
4. hyperparameter tuning (~ 1 week)

"""

"""
# test bleu score
resp = word_tokenize("Oh no! I'm so sorry to hear that. I hope you feel better soon.")
ref = ['Ouch', 'that', 'sucks', '.', 'Honestly', ',', 'if', 'you', 'had', 'a', 'crappy', 'obamacare', 'bronze', 'plan', 'its', 'unlikely', 'they', "'d", 'pay', 'for', 'anything', 'anyways', '.', 'Its', 'always', 'a', 'nightmare', '.', 'You', "'ve", 'got', 'to', 'spend', 'a', 'fair', 'amont', 'of', 'cash', 'before', 'they', 'cover', 'anything', 'but', 'a', 'checkup', '.']
score1 = nltk.translate.bleu_score.sentence_bleu([ref], resp, weights=(1, 0, 0))
score2 = nltk.translate.bleu_score.sentence_bleu([ref], resp, weights=(0, 1, 0))
score3 = nltk.translate.bleu_score.sentence_bleu([ref], resp, weights=(0, 0, 1))
ngram_score_list = [score1, score2, score3]
score = sum(ngram_score_list) / len(ngram_score_list)
print(score)
#print(0.25*score)
#print(weighted_score)
"""