import os
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM, \
    RobertaTokenizerFast, RobertaForSequenceClassification, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
import torch
import nltk
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from scipy.special import softmax, logit
from helper import weighted_bleu_score, get_js_distance, emo_dis_bleu, append_scores

text_generation_model = "ppo_model" # blenderbot, ppo_model
ppo_model = "../DEV-mimic-high-reward-blenderbot-400m-emo-probi-bleu-last-score0.7700437359931219-bleu0.06567799299955368"

output_path_prefix = 'DEV_ppo_test'
load_path_prefix = '../'

device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
#device = "mps"

# evaluation models
emo_model_id = "SamLowe/roberta-base-go_emotions"
emo_tokenizer = AutoTokenizer.from_pretrained(emo_model_id)
emo_model = AutoModelForSequenceClassification.from_pretrained(emo_model_id, torch_dtype=torch.float32).to(device)
emo_classifier = pipeline('text-classification', model=emo_model_id, tokenizer=emo_model_id, max_length=512, truncation=True, top_k=None)

emp_classifier_model = f"{load_path_prefix}models/roberta-empathy-03-06-2023-18_21_58"
empathy_model_id = emp_classifier_model
empathy_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
empathy_model = RobertaForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32).to(device)
empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True)
#empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True, device=0)


if(os.path.exists(f'{load_path_prefix}modeldata/emo_probi.p')):
    print("LOADING data emotion probability distribution...")
    with open(f'{load_path_prefix}modeldata/emo_probi.p', "rb") as f:
        [all_emo_probi, _] = pickle.load(f)
    f.close()
all_emo_probi = dict(all_emo_probi)


# test model
BLEU_score_list = []
texts = []


if text_generation_model == "opt-iml-1.3b":
    pretrained_model = "facebook/opt-iml-1.3b"
    model_id = pretrained_model
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)
elif text_generation_model == "ppo_model":
    model_id = ppo_model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)
else:
    pretrained_model = "facebook/blenderbot-400M-distill"
    model_id = pretrained_model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

with open(f"{load_path_prefix}modeldata/test_dialogue_dataset.p", "rb") as f:
    [test_dataset] = pickle.load(f)
#test_dataset = Dataset.from_dict(test_dataset)[:10]
test_dataset = Dataset.from_dict(test_dataset).shuffle(seed=2023).select(range(5))
#test_dataset = Dataset.from_dict(test_dataset)

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
#emp_score_label_weight = [-1, 1, 4]
mean_bleu = 0
emp_ratio = []
try:
    with open(f'{output_path_prefix}_text_log_emo_probi_score.txt', 'w') as text_log:
        counter = 0
        prompts = []
        query_list = []
        score_list = []
        for test_query in test_dataset:
            input_texts = test_query["prompt"]
            prompts.append(input_texts.replace("</s>", "").replace("_comma_", ","))
            query_list.append(test_query["query"])
            target = test_query["query"]["target"]
            #print(target)
            #if counter > 590:
            #    text_log.write(f"{counter} tokenizing... \n")
            input_ids = tokenizer(input_texts, return_tensors="pt", padding='max_length', max_length=128, truncation=True).to(device)
            #input_ids = test_query['input_ids']
            #if counter > 590:
            #    text_log.write(f"{counter} generating response... \n")
            outputs = model.generate(**input_ids, do_sample=True, max_new_tokens=40, use_cache=True)
            #outputs = model.generate(**input_ids, num_beams=5, do_sample=True, max_new_tokens=40, use_cache=True, num_return_sequences=1)
            #if counter > 590:
            #    text_log.write(f"{counter} decoding response... \n")
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            texts.append(generated_texts[0])
            input_texts = input_texts.replace("_comma_", ",")
            text_log.write(f"{counter} Prompt: {input_texts} \n")
            print(f"{counter} Prompt: {input_texts} \n")
            text_log.write(f"{counter} Response: {generated_texts[0]} \n")
            print(f"{counter} Response: {generated_texts[0]} \n")
            text_log.write(f"{counter} Ref: {target} \n \n")
            print(f"{counter} Ref: {target} \n \n")
            counter += 1

            # Calculate bleu score
            test_response = word_tokenize(generated_texts[0])
            dev_target = word_tokenize(test_query["target"])
            dev_BLEUscore = weighted_bleu_score(dev_target, test_response)
            BLEU_score_list.append(dev_BLEUscore)
            text_log.write(f"{counter} BLEU: {dev_BLEUscore}\n")

        mean_bleu = sum(BLEU_score_list) / len(BLEU_score_list)

        # calculate emo distribution
        prompt_results = emo_classifier(prompts)
        emo_results = emo_classifier(texts)
        list_emo_score, mean_emo_score = get_js_distance(prompt_results, emo_results)
        text_log.write(f"{counter} Mean Emo Score: {mean_emo_score}\n")

        for i in range(len(BLEU_score_list)):
            temp_score = (list_emo_score[i] * emp_weight) + (BLEU_score_list[i] * fluency_weight)
            score_list.append(np.float32(temp_score))

        #score_list, _, _ = emo_dis_bleu(query_list, texts, prompt_results, emo_results, weights=[emp_weight, fluency_weight])
        # take mean
        current_score = sum(score_list) / len(score_list)
        text_log.write(f"{counter} Current weighted Score: {current_score}\n")

        # get empathy label and score from empathy classifier
        text_log.write(f"{counter} Get empathy labels. \n")
        #texts = texts.to(device)
        #print(device)
        emp_results = empathy_classifier(texts, padding='max_length', truncation=True, max_length=512)
        text_log.write(f"{counter} Obtained results. \n")
        emp_results = Dataset.from_list(emp_results)
        emp_results_labels = emp_results['label']

        #print(emp_results_labels)
        label_number = {label: emp_results_labels.count(label) for label in emp_results_labels}
        #print(label_number)
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

        text_log.write(f"{counter} Empathy Ratio: no empathy {emp_ratio[0]}, weak empathy {emp_ratio[1]}, strong empathy {emp_ratio[2]}\n")

        text_log.write(f"Mean BLEU: {mean_bleu}, emp ratio: {emp_ratio}, score: {current_score}. \n")
    text_log.close()

    with open(f'{output_path_prefix}_test_score_log_emo_probi_score.txt', 'w') as score_log:
        score_log.write(f"Mean BLEU of this model: {mean_bleu}. \n")
        #score_log.write(f"Emo distribution similarity of this model: {emo_score}. \n")
        score_log.write(f"Empathy Ratio, no empathy: {emp_ratio[0]}, weak empathy {emp_ratio[1]}, strong empathy: {emp_ratio[2]}\n")
        score_log.write(f"Score of this model: {current_score}. \n")
        print(f"Mean BLEU of this model: {mean_bleu}. \n")
        #print(f"Emo distribution similarity of this model: {emo_score}. \n")
        print(f"Empathy Ratio: no empathy {emp_ratio[0]}, weak empathy {emp_ratio[1]}, strong empathy {emp_ratio[2]}\n")
        print(f"Score of this model: {current_score}. \n")
    score_log.close()
except Exception as err:
    with open(f'{output_path_prefix}_test_error_log_emo_probi_score.txt', 'w') as err_log:
        err_log.write(f"Unexpected {err=}, {type(err)=}")
    err_log.close()