import matplotlib
import os
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from datasets import Dataset
import torch
from nltk.tokenize import word_tokenize
import nltk
from scipy.spatial import distance
from scipy.special import softmax

path_prefix = 'DEV_debug_test'
emo_model_id = "SamLowe/roberta-base-go_emotions"
emo_classifier = pipeline('text-classification', model=emo_model_id,tokenizer=emo_model_id, max_length=512, truncation=True, top_k=None)

if(os.path.exists('../modeldata/emo_probi.p')):
    print("LOADING data emotion probability distribution...")
    with open('../modeldata/emo_probi.p', "rb") as f:
        [all_emo_probi, _] = pickle.load(f)
    f.close()
all_emo_probi = dict(all_emo_probi)


def append_scores(labels, original, sample):
    all_emo_scores = original
    for sam in sample:
        for s in sam:
            emo = s.get('label')
            prev_score = original[emo]
            score = s.get('score')
            all_emo_scores[emo] = (prev_score + score)
    all_scores = list(zip(*all_emo_scores.items()))[1]
    probi = softmax(all_scores)
    all_emo_scores = dict(zip(labels, probi))
    return all_emo_scores


def weighted_bleu_score(target, response):
    score1 = nltk.translate.bleu_score.sentence_bleu([target], response, weights=(1, 0, 0))
    score2 = nltk.translate.bleu_score.sentence_bleu([target], response, weights=(0, 1, 0))
    score3 = nltk.translate.bleu_score.sentence_bleu([target], response, weights=(0, 0, 1))
    ngram_score_list = [score1, score2, score3]
    return (sum(ngram_score_list) / len(ngram_score_list))


device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
BLEU_score_list = []
texts = []
ppo_model = "../DEV-blenderbot-400m-emo-probi-bleu-epoch0-score0.382-bleu0.06629"
model_id = ppo_model
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

with open("../modeldata/test_dialogue_dataset.p", "rb") as f:
    [test_dataset] = pickle.load(f)
#test_dataset = Dataset.from_dict(test_dataset)[:100]
test_dataset = Dataset.from_dict(test_dataset)

def tokenize(sample):
    prompt = sample["prompt"] # prompt
    continuation = sample["target"] # utterance

    sample["input_ids"] = tokenizer.encode(prompt)
    sample["query"] = {"prompt": tokenizer.decode(sample["input_ids"]), "target": continuation}
    return sample

test_dataset = test_dataset.map(tokenize, batched=False)
test_dataset.set_format(type="torch")
emp_weight = 0.2
fluency_weight = 0.8

#print("Start testing...")
#try:
with open(f'{path_prefix}_text_log_emo_probi_score.txt', 'w') as text_log:
    counter = 0
    for test_query in test_dataset:
        input_texts = test_query["prompt"]
        target = test_query["query"]["target"]
        #print(target)
        input_ids = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
        #input_ids = test_query['input_ids']
        outputs = model.generate(**input_ids, do_sample=True, max_new_tokens=40, use_cache=True)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        texts.append(generated_texts[0])
        text_log.write(f"{counter} Prompt: {input_texts} \n")
        text_log.write(f"{counter} Response: {generated_texts[0]} \n")
        text_log.write(f"{counter} Ref: {target} \n \n")
        counter += 1

        # Calculate bleu score
        test_response = word_tokenize(generated_texts[0])
        dev_target = word_tokenize(test_query["target"])
        dev_BLEUscore = weighted_bleu_score(dev_target, test_response)
        BLEU_score_list.append(dev_BLEUscore)

    mean_bleu = sum(BLEU_score_list) / len(BLEU_score_list)

    # calculate emo distribution
    emo_results = emo_classifier(texts)
    labels = [s.get('label') for s in emo_results[0]]
    zeros = [0] * len(labels)
    score_dict = dict(zip(labels, zeros))
    empathy_results = append_scores(labels, score_dict, emo_results)
    # sort alphabetically
    empathy_results = dict(sorted(empathy_results.items(), key=lambda x: x[0].lower()))
    all_emo_probi_values = list(all_emo_probi.values())
    empathy_results_values = list(empathy_results.values())

    js_distance = distance.jensenshannon(all_emo_probi_values, empathy_results_values)
    # js_distance: identical = 0, entirely different = 1, reverse this for reward
    emo_score = js_distance

    current_score = (emo_score * emp_weight) + ((1 - mean_bleu) * fluency_weight)
    text_log.write(f"Mean BLEU: {mean_bleu}, emo distribution similarity: {emo_score}, score: {current_score}. \n")
text_log.close()

with open(f'{path_prefix}_test_score_log_emo_probi_score.txt', 'w') as score_log:
    score_log.write(f"Mean BLEU of this model: {mean_bleu}. \n")
    score_log.write(f"Emo distribution similarity of this model: {emo_score}. \n")
    score_log.write(f"Score of this model: {current_score}. \n")
    print(f"Mean BLEU of this model: {mean_bleu}. \n")
    print(f"Emo distribution similarity of this model: {emo_score}. \n")
    print(f"Score of this model: {current_score}. \n")
score_log.close()
#except Exception as err:
#    with open(f'{path_prefix}_test_error_log_emo_probi_score.txt', 'w') as err_log:
#        err_log.write(f"Unexpected {err=}, {type(err)=}")
#    err_log.close()