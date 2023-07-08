import nltk
import torch
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from scipy.special import softmax, logit
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, RobertaForSequenceClassification, \
    RobertaTokenizerFast


def load_emo_classifier():
    emo_model_id = "SamLowe/roberta-base-go_emotions"
    emo_tokenizer = AutoTokenizer.from_pretrained(emo_model_id)
    emo_model = AutoModelForSequenceClassification.from_pretrained(emo_model_id, torch_dtype=torch.float32).to(device)
    emo_classifier = pipeline('text-classification', model=emo_model_id, tokenizer=emo_model_id, max_length=512,
                              truncation=True, top_k=None)
    return emo_classifier
def load_empathy_classifier(path_prefix=""):
    empathy_model_id = f"{path_prefix}models/roberta-empathy-03-06-2023-18_21_58"
    empathy_model = RobertaForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32)
    empathy_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    empathy_classifier = pipeline('text-classification', model=empathy_model_id, tokenizer=empathy_tokenizer,
                                  max_length=512, truncation=True)
    return empathy_classifier
def load_toxicity_classifier():
    toxicity_model_id = "martin-ha/toxic-comment-model"
    toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_id)
    toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_id)
    toxicity_classifier = pipeline('text-classification', model=toxicity_model, tokenizer=toxicity_tokenizer,
                                   top_k=None)
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

def emo_dis_ppl_toxic(prompt_results, emo_results, inverse_perplexity, toxicity, weights=[0.2, 0.7, 0.1]):
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