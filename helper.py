import nltk
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from scipy.special import softmax, logit
import numpy as np

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


def emo_dis_bleu(batch_query, batch_response, prompt_results, emo_results, weights=[0.2, 0.8]):
    emo_weight = weights[0]
    fluency_weight = weights[1]
    score_list = []
    BLEUscore_list = []
    
    list_emo_score, mean_emo_score = get_js_distance(prompt_results, emo_results)

    for i in range(len(batch_response)):
        temp_score = 0
        response = word_tokenize(batch_response[i])
        target = batch_query[i].get("target").replace("_comma_", ",")
        target = word_tokenize(target)
        # Compute BLEU score
        BLEUscore = weighted_bleu_score(target, response)
        BLEUscore_list.append(BLEUscore)

        emp_score = list_emo_score[i]
        # better response higher score
        temp_score = (emp_score * emo_weight) + (BLEUscore * fluency_weight)
        score_list.append(np.float32(temp_score))

    return score_list, list_emo_score, BLEUscore_list