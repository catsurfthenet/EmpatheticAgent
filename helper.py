import os
import pickle

import nltk
import torch
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from scipy.special import softmax, logit
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, RobertaForSequenceClassification, \
    RobertaTokenizerFast
from datasets import Dataset
from trl.core import LengthSampler
import torch.nn.utils.rnn as rnn_utils

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset_no_token(
    config, dataset_path='modeldata/dialogue_dataset.p', input_min_text_length=5, input_max_text_length=100, size=-1
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    emo_labels = ['sadness', 'disappointment', 'neutral', 'fear', 'nervousness', 'disapproval', 'realization',
                  'annoyance', 'grief', 'approval', 'caring', 'remorse', 'disgust', 'desire', 'love', 'anger',
                  'embarrassment', 'joy', 'admiration', 'relief', 'surprise', 'optimism', 'confusion', 'curiosity',
                  'amusement', 'excitement', 'gratitude', 'pride']
    emo_labels = [f"[{i}]" for i in emo_labels]
    special_tokens_dict = {'additional_special_tokens': emo_labels}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    for i in emo_labels:
        tok_id = tokenizer.convert_tokens_to_ids(i)
    """
    #ds = load_dataset(dataset_name, split="train")
    if (os.path.exists(dataset_path)):
        print("LOADING empathetic_dialogue")
        with open(dataset_path, "rb") as f:
            [data] = pickle.load(f)
    ds = Dataset.from_dict(data)
    if size > -1:
        ds = ds.shuffle(seed=2024).select(range(size))

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def format(sample):
        prompt = sample["prompt"] # prompt
        continuation = sample["target"] # utterance
        target_emotion = sample["target_emo"]

        #sample["input_ids"] = tokenizer.encode_plus(prompt, add_special_tokens=True)[: input_size()]
        #sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
        #tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
        #sample["input_ids"] += [2] * max((128 - len(sample["input_ids"])), 0)
        #sample["target_ids"] = tokenizer.encode(continuation)[: input_size()]
        sample["input_ids"] = sample["prompt"]
        sample["query"] = {"prompt": prompt, #tokenizer.decode(sample["input_ids"]),
                           "target": continuation,
                           "target_emo": target_emotion}
        return sample

    ds = ds.map(format, batched=False)
    ds.set_format(type="torch")

    #ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]
    return ds

def build_dataset(
    config, dataset_path='modeldata/dialogue_dataset.p', input_min_text_length=5, input_max_text_length=100, size=-1
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    emo_labels = ['sadness', 'disappointment', 'neutral', 'fear', 'nervousness', 'disapproval', 'realization',
                  'annoyance', 'grief', 'approval', 'caring', 'remorse', 'disgust', 'desire', 'love', 'anger',
                  'embarrassment', 'joy', 'admiration', 'relief', 'surprise', 'optimism', 'confusion', 'curiosity',
                  'amusement', 'excitement', 'gratitude', 'pride']
    emo_labels = [f"[{i}]" for i in emo_labels]
    special_tokens_dict = {'additional_special_tokens': emo_labels}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    for i in emo_labels:
        tok_id = tokenizer.convert_tokens_to_ids(i)

    #ds = load_dataset(dataset_name, split="train")
    if (os.path.exists(dataset_path)):
        print("LOADING empathetic_dialogue")
        with open(dataset_path, "rb") as f:
            [data] = pickle.load(f)
    ds = Dataset.from_dict(data)
    if size > -1:
        ds = ds.shuffle(seed=2023).select(range(size))

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"] # prompt
        continuation = sample["target"] # utterance

        sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
        #sample["input_ids"] += [0] * max((128 - len(sample["input_ids"])), 0)
        #sample["target_ids"] = tokenizer.encode(continuation)[: input_size()]
        sample["query"] = {"prompt": tokenizer.decode(sample["input_ids"]), "target": continuation}
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]
    return ds


def padding(data):
    padded = rnn_utils.pad_sequence(data)
    padded = list(map(torch.Tensor, padded.T))
    return padded

def build_pad_dataset(
    config, dataset_path='modeldata/dialogue_dataset.p', input_min_text_length=5, input_max_text_length=100, size=-1
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    emo_labels = ['sadness', 'disappointment', 'neutral', 'fear', 'nervousness', 'disapproval', 'realization',
                  'annoyance', 'grief', 'approval', 'caring', 'remorse', 'disgust', 'desire', 'love', 'anger',
                  'embarrassment', 'joy', 'admiration', 'relief', 'surprise', 'optimism', 'confusion', 'curiosity',
                  'amusement', 'excitement', 'gratitude', 'pride']
    emo_labels = [f"[{i}]" for i in emo_labels]
    special_tokens_dict = {'additional_special_tokens': emo_labels}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    for i in emo_labels:
        tok_id = tokenizer.convert_tokens_to_ids(i)

    #ds = load_dataset(dataset_name, split="train")
    if (os.path.exists(dataset_path)):
        print("LOADING empathetic_dialogue")
        with open(dataset_path, "rb") as f:
            [data] = pickle.load(f)
    ds = Dataset.from_dict(data)
    if size > -1:
        ds = ds.shuffle(seed=2023).select(range(size))

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"] # prompt
        continuation = sample["target"] # utterance

        sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
        sample["input_ids"] += [2] * max((128 - len(sample["input_ids"])), 0)
        #sample["target_ids"] = tokenizer.encode(continuation)[: input_size()]
        sample["query"] = {"prompt": tokenizer.decode(sample["input_ids"]),
                           "target": continuation}
        return sample

    ds = ds.map(tokenize, batched=False)
    #og = ds["input_ids"]
    #ds_ids = padding(torch.tensor(ds["input_ids"]))
    #ds["input_ids"] = ds_ids

    ds.set_format(type="torch")

    #ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds

def build_train_dataset(
    config, dataset_path='modeldata/dialogue_dataset.p', input_min_text_length=5, input_max_text_length=100, size=-1
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    emo_labels = ['sadness', 'disappointment', 'neutral', 'fear', 'nervousness', 'disapproval', 'realization',
                  'annoyance', 'grief', 'approval', 'caring', 'remorse', 'disgust', 'desire', 'love', 'anger',
                  'embarrassment', 'joy', 'admiration', 'relief', 'surprise', 'optimism', 'confusion', 'curiosity',
                  'amusement', 'excitement', 'gratitude', 'pride']
    emo_labels = [f"[{i}]" for i in emo_labels]
    special_tokens_dict = {'additional_special_tokens': emo_labels}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    for i in emo_labels:
        tok_id = tokenizer.convert_tokens_to_ids(i)

    #ds = load_dataset(dataset_name, split="train")
    if (os.path.exists(dataset_path)):
        print("LOADING empathetic_dialogue")
        with open(dataset_path, "rb") as f:
            [data] = pickle.load(f)
    ds = Dataset.from_dict(data)
    if size > -1:
        ds = ds.shuffle(seed=2024).select(range(size))

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"] # prompt
        continuation = sample["target"] # utterance
        target_emotion = sample["target_emo"]

        #sample["input_ids"] = tokenizer.encode_plus(prompt, add_special_tokens=True)[: input_size()]
        sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
        #tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
        #sample["input_ids"] += [2] * max((128 - len(sample["input_ids"])), 0)
        #sample["target_ids"] = tokenizer.encode(continuation)[: input_size()]
        sample["query"] = {"prompt": tokenizer.decode(sample["input_ids"]),
                           "target": continuation,
                           "target_emo": target_emotion}
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    #ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]
    return ds

def build_pretrain_dataset(
    config, dataset_path='modeldata/dialogue_dataset.p', input_min_text_length=5, input_max_text_length=100, size=-1
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    #tokenizer.pad_token = tokenizer.eos_token
    emo_labels = ['sadness', 'disappointment', 'neutral', 'fear', 'nervousness', 'disapproval', 'realization',
                  'annoyance', 'grief', 'approval', 'caring', 'remorse', 'disgust', 'desire', 'love', 'anger',
                  'embarrassment', 'joy', 'admiration', 'relief', 'surprise', 'optimism', 'confusion', 'curiosity',
                  'amusement', 'excitement', 'gratitude', 'pride']
    emo_labels = [f"[{i}]" for i in emo_labels]
    special_tokens_dict = {'additional_special_tokens': emo_labels}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    for i in emo_labels:
        tok_id = tokenizer.convert_tokens_to_ids(i)

    #ds = load_dataset(dataset_name, split="train")
    if (os.path.exists(dataset_path)):
        print("LOADING empathetic_dialogue")
        with open(dataset_path, "rb") as f:
            [data] = pickle.load(f)
    ds = Dataset.from_dict(data)
    if size > -1:
        ds = ds.shuffle(seed=2023).select(range(size))

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"] # prompt
        continuation = sample["target"] # utterance
        sample["input_ids"] = tokenizer(prompt, add_special_tokens=True, padding='max_length', max_length=128, truncation=True)["input_ids"] #, padding='max_length', max_length=128

        #sample["input_ids"] = tokenizer.encode_plus(prompt, add_special_tokens=True)["input_ids"][: input_size()]
        #sample["input_ids"] += [0] * max((128 - len(sample["input_ids"])), 0)
        #sample["target_ids"] = tokenizer.encode(continuation)[: input_size()]
        sample["query"] = {"prompt": tokenizer.batch_decode(sample["input_ids"]),
                           "target": continuation}
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    #ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]
    return ds

def get_mean(scores):
    return (sum(scores) / len(scores))

def get_bertscore_results(bertscore_results):
    precision = get_mean(bertscore_results["precision"])
    recall = get_mean(bertscore_results["recall"])
    f1 = get_mean(bertscore_results["f1"])
    return precision, recall, f1

def get_bertscore_results_batch(bertscore_results):
    precision = get_mean([get_mean(b) for b in bertscore_results["precision"]])
    recall = get_mean([get_mean(b) for b in bertscore_results["recall"]])
    f1 = get_mean([get_mean(b) for b in bertscore_results["f1"]])
    return precision, recall, f1
def load_emo_classifier(device):
    emo_model_id = "SamLowe/roberta-base-go_emotions"
    emo_tokenizer = AutoTokenizer.from_pretrained(emo_model_id)
    emo_model = AutoModelForSequenceClassification.from_pretrained(emo_model_id, torch_dtype=torch.float32).to(device)
    emo_classifier = pipeline('text-classification', model=emo_model_id, tokenizer=emo_model_id, max_length=512,
                              truncation=True, top_k=None)
    return emo_classifier

def load_empathy_classifier(path_prefix="", cuda=False):
    empathy_model_id = f"{path_prefix}models/roberta-empathy-03-06-2023-18_21_58"
    empathy_model = RobertaForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32)
    empathy_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    if cuda:
        empathy_classifier = pipeline('text-classification', model=empathy_model_id, tokenizer=empathy_tokenizer,
                                      max_length=512, truncation=True, device=0)
    else:
        empathy_classifier = pipeline('text-classification', model=empathy_model_id, tokenizer=empathy_tokenizer,
                                  max_length=512, truncation=True)
    return empathy_classifier
def load_toxicity_classifier(cuda=False):
    toxicity_model_id = "martin-ha/toxic-comment-model"
    toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_id)
    toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_id)
    if cuda:
        toxicity_classifier = pipeline('text-classification', model=toxicity_model, tokenizer=toxicity_tokenizer,
                                       top_k=None, device=0)
    else:
        toxicity_classifier = pipeline('text-classification', model=toxicity_model, tokenizer=toxicity_tokenizer)
    return toxicity_classifier

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


def get_js_distance(prompt_results, emo_results):
    labels = [s.get('label') for s in emo_results[0]]
    zeros = [0] * len(labels)
    list_js_distance = []
    for i in range(len(prompt_results)):
        prompt_dict = dict(zip(labels, zeros))
        response_dict = dict(zip(labels, zeros))
        for j in range(len(prompt_results[i])):
            label = prompt_results[i][j].get("label")
            prompt_dict[label] = prompt_results[i][j].get("score")
            label = emo_results[i][j].get("label")
            response_dict[label] = emo_results[i][j].get("score")

        prompt_value = dict(sorted(prompt_dict.items(), key=lambda x: x[0].lower())).values()
        prompt_value = list(prompt_value)
        response_value = dict(sorted(response_dict.items(), key=lambda x: x[0].lower())).values()
        response_value = list(response_value)
        # js_distance: identical = 0, entirely different = 1
        js_distance = distance.jensenshannon(prompt_value, response_value)
        # higher score for higher similarity
        list_js_distance.append((1 - js_distance))

    mean_js_distance = sum(list_js_distance) / len(list_js_distance)
    return list_js_distance, mean_js_distance

def get_emo_counts(prompt_results, emo_results, top=10):
    list_labels = []
    #label_counts = []
    #zeros = [0] * len(labels)
    list_score = []
    for i in range(len(prompt_results)):
        labels = []
        for j in range(top):
            prompt_label = prompt_results[i][j].get("label")
            resp_label = emo_results[i][j].get("label")
            labels.append(prompt_label)
            labels.append(resp_label)
        list_labels.append(labels)
    for labels in list_labels:
        label_counts = len(set(labels))
        if label_counts == top:
            list_score.append(1 - 1e-10)
        elif label_counts > top:
            score = (top - (label_counts - top)) / top
            score = 1e-10 if (score == 0) else score
            list_score.append(score) # counts how many more that are matched
    return np.float32(list_score), get_mean(list_score)


def emo_dis_ppl_toxic(prompt_results, emo_results, inverse_perplexity, toxicity, weights=[0.4, 0.4, 0.2]):
    emo_weight = weights[0]
    toxicity_weight = weights[1]
    fluency_weight = weights[2]
    score_list = []
    weighted_ppl = inverse_perplexity * fluency_weight
    list_emo_score = [0] * len(prompt_results)
    mean_emo_score, mean_toxic_score = 0, 0
    toxicity_score = []
    if emo_weight > 0:
        list_emo_score, mean_emo_score = get_js_distance(prompt_results, emo_results)

    for i in range(len(list_emo_score)):
        if toxicity_weight > 0:
            if toxicity[i]["label"] == "toxic":
                toxic_score = 1e-10
            else:
                toxic_score = toxicity[i]["score"]
        else:
            toxic_score = 1e-10
        toxicity_score.append(toxic_score)
        emp_score = list_emo_score[i]
        # better response higher score
        temp_score = (emp_score * emo_weight) + (weighted_ppl) + (toxicity_weight * toxic_score)
        score_list.append(np.float32(temp_score))

    mean_toxic_score = get_mean(toxicity_score)
    return score_list, list_emo_score, mean_emo_score, mean_toxic_score

def emo_count_ppl_toxic(prompt_results, emo_results, inverse_perplexity, toxicity, weights=[0.4, 0.4, 0.2]):
    emo_weight = weights[0]
    toxicity_weight = weights[1]
    fluency_weight = weights[2]
    score_list = []
    weighted_ppl = inverse_perplexity * fluency_weight
    list_emo_score = [0] * len(prompt_results)
    mean_emo_score, mean_toxic_score = 0, 0
    toxicity_score = []
    if emo_weight > 0:
        list_emo_score, mean_emo_score = get_emo_counts(prompt_results, emo_results)

    for i in range(len(list_emo_score)):
        if toxicity_weight > 0:
            if toxicity[i]["label"] == "toxic":
                toxic_score = 1e-10
            else:
                toxic_score = toxicity[i]["score"]
        else:
            toxic_score = 1e-10
        toxicity_score.append(toxic_score)
        emp_score = list_emo_score[i]
        # better response higher score
        temp_score = (emp_score * emo_weight) + (weighted_ppl) + (toxicity_weight * toxic_score)
        score_list.append(np.float32(temp_score))

    mean_toxic_score = get_mean(toxicity_score)
    return score_list, list_emo_score, mean_emo_score, mean_toxic_score

def emo_count_ppl_ref_emo(prompt_results, emo_results, inverse_perplexity, toxicity, weights=[0.4, 0.4, 0.2]):
    emo_weight = weights[0]
    toxicity_weight = weights[1]
    fluency_weight = weights[2]
    score_list = []
    weighted_ppl = inverse_perplexity * fluency_weight
    list_emo_score = [0] * len(prompt_results)
    mean_emo_score, mean_toxic_score = 0, 0
    toxicity_score = []
    if emo_weight > 0:
        list_emo_score, mean_emo_score = get_emo_counts(prompt_results, emo_results)

    for i in range(len(list_emo_score)):
        if toxicity_weight > 0:
            if toxicity[i]["label"] == "toxic":
                toxic_score = 1e-10
            else:
                toxic_score = toxicity[i]["score"]
        else:
            toxic_score = 1e-10
        toxicity_score.append(toxic_score)
        emp_score = list_emo_score[i]
        # better response higher score
        temp_score = (emp_score * emo_weight) + (weighted_ppl) + (toxicity_weight * toxic_score)
        score_list.append(np.float32(temp_score))

    mean_toxic_score = get_mean(toxicity_score)
    return score_list, list_emo_score, mean_emo_score, mean_toxic_score


def emo_dis_ppl(prompt_results, emo_results, inverse_perplexity, weights=[0.2, 0.8]):
    emo_weight = weights[0]
    fluency_weight = weights[1]
    score_list = []
    weighted_ppl = inverse_perplexity * fluency_weight
    list_emo_score = [0] * len(prompt_results)
    if emo_weight > 0:
        list_emo_score, mean_emo_score = get_js_distance(prompt_results, emo_results)

    for i in range(len(list_emo_score)):
        emp_score = list_emo_score[i]
        # better response higher score
        temp_score = (emp_score * emo_weight) + (weighted_ppl)
        score_list.append(np.float32(temp_score))

    return score_list, list_emo_score, mean_emo_score

def emo_count_ppl(prompt_results, emo_results, inverse_perplexity, weights=[0.2, 0.8]):
    emo_weight = weights[0]
    fluency_weight = weights[1]
    score_list = []
    weighted_ppl = inverse_perplexity * fluency_weight
    list_emo_score = [0] * len(prompt_results)
    if emo_weight > 0:
        list_emo_score, mean_emo_score = get_js_distance(prompt_results, emo_results)

    for i in range(len(list_emo_score)):
        emp_score = list_emo_score[i]
        # better response higher score
        temp_score = (emp_score * emo_weight) + (weighted_ppl)
        score_list.append(np.float32(temp_score))

    return score_list, list_emo_score, mean_emo_score


def emo_dis_bleu(batch_query, batch_response, prompt_results, emo_results, weights=[0.2, 0.8]):
    emo_weight = weights[0]
    fluency_weight = weights[1]
    score_list = []
    BLEUscore_list = []
    list_emo_score = [0] * len(batch_response)
    if emo_weight > 0:
        list_emo_score, mean_emo_score = get_js_distance(prompt_results, emo_results)

    for i in range(len(batch_response)):
        temp_score = 0
        response = word_tokenize(batch_response[i])
        target = batch_query[i].get("target").replace("_comma_", ",")
        target = word_tokenize(target)
        # Compute BLEU score
        BLEUscore = weighted_bleu_score(target, response)
        if BLEUscore == 0:
            BLEUscore += 1e-10 # so that logit != inf
        elif BLEUscore == 1:
            BLEUscore -= 1e-10
        BLEUscore_list.append(BLEUscore)

        emp_score = list_emo_score[i]
        # better response higher score
        temp_score = (emp_score * emo_weight) + (BLEUscore * fluency_weight)
        score_list.append(np.float32(temp_score))

    return score_list, list_emo_score, BLEUscore_list

def get_empathy_ratio(emp_results):
    emp_results = Dataset.from_list(emp_results)
    emp_results_labels = emp_results['label']

    label_number = {label: emp_results_labels.count(label) for label in emp_results_labels}
    label_names = list(zip(*label_number.items()))[0]

    if len(label_names) > 3:
        label0 = label_number['LABEL_0'] / len(emp_results_labels)
        label1 = label_number['LABEL_1'] / len(emp_results_labels)
        label2 = label_number['LABEL_2'] / len(emp_results_labels)
    else:
        label0, label1, label2 = 0, 0, 0
        for lab in label_names:
            # nothing to be added for LABEL_0
            if lab == 'LABEL_0':
                label0 = label_number[lab] / len(emp_results_labels)
            elif lab == 'LABEL_1':
                label1 = label_number[lab] / len(emp_results_labels)
            elif lab == 'LABEL_2':
                label2 = label_number[lab] / len(emp_results_labels)
    emp_ratio = [label0, label1, label2]

    return emp_ratio

def get_FACE_loss(token_freq, outputs, special_ids, loss_logits):
    weights_RF = get_RF(token_freq)
    # normalise
    weights_RF = weights_RF / sum(weights_RF)
    # weights_RF = weights_RF * len(weights_RF) # make mean = 1
    div_loss = 0
    for i in range(len(outputs)):
        # logits_wo_sp_tokens = loss_logits[i][0]
        tensor_size = len(outputs[i]) if len(outputs[i]) < len(loss_logits[i]) else len(loss_logits[i])
        for o in range(tensor_size):
            if outputs[i][o] in special_ids:  # skip special tokens
                continue
            token = outputs[i][o]
            if (token - 4) >= 0:  # assert no indexing error
                #a = loss_logits[i][o][token]
                #b = weights_RF[token - 4]
                div_loss += loss_logits[i][o][token] * weights_RF[token - 4] #account for offset

    return div_loss


def get_FACE_reward(token_freq, outputs, special_ids):
    weights_RF = get_RF(token_freq)
    # normalise
    weights_RF = weights_RF / sum(weights_RF)
    # weights_RF = weights_RF * len(weights_RF) # make mean = 1
    div_reward = 0
    for i in range(len(outputs)):
        # logits_wo_sp_tokens = loss_logits[i][0]
        tensor_size = len(outputs[i])
        for o in range(tensor_size):
            if outputs[i][o] in special_ids:  # skip special tokens
                continue
            token = outputs[i][o]
            if (token - 4) >= 0:  # assert no indexing error
                div_reward += weights_RF[token - 4] #account for offset
    return div_reward


def get_RF(token_freq):
    current_freq = np.array(list(token_freq.values())[4:8008])  # 0-4, 8008+ are sp tokens
    relative_freq = current_freq / sum(current_freq)
    max_RF = max(relative_freq)
    weights_RF = (-1 / max_RF) * relative_freq + 1
    return weights_RF