import os
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM, \
    RobertaTokenizerFast, RobertaForSequenceClassification, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset
import torch
from evaluate import load
import nltk
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from scipy.special import softmax, logit
from helper import get_mean, compute_freq, get_bertscore_results, get_bertscore_results_batch
from torch.utils.data import DataLoader
from promcse import PromCSE

# define which model to evaluate
#change
finetune = True
load_path_prefix = ''
ppo_model = "no_sp_token_50epochs-2500-loss1.43364" #"w1-1-1-1-0_50epochs-4250-loss1.32314"
text_generation_model = "ppo_model" # blenderbot, ppo_model, dialogpt, opt_iml

# set path prefix
output_path_prefix = f"{ppo_model}_full_test"#f"FTM15-8_w5215_full_local_LLR_ED_ts2000_epoch10_sim_loss_emo_last_{text_generation_model}_bert_test" #'DEV_cont3_emo_toxic-w6-4-0_bert_test'
load_path_prefix = '' # change

device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
negative_sample_size = 8
#device = "mps"

# load evaluation models
# load empathy classification model
emp_classifier_model = f"{load_path_prefix}models/roberta-empathy-03-06-2023-18_21_58"
empathy_model_id = emp_classifier_model
empathy_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
empathy_model = RobertaForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32).to(device)
# change
#empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True)
empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True, device=0)

# load BertSCORE
bertscore = load("bertscore")

# load PromCSE
promcse_model = PromCSE("YuxinJiang/unsup-promcse-bert-base-uncased", "cls_before_pooler", 16)

# get trained model to be evaluated
if text_generation_model == "opt_iml":
    pretrained_model = "facebook/opt-iml-1.3b"
    model_id = pretrained_model
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)
elif text_generation_model == "dialogpt":
    pretrained_model = "microsoft/DialoGPT-small"
    model_id = pretrained_model
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)
elif text_generation_model == "gpt_neo":
    pretrained_model = "EleutherAI/gpt-neo-125m"
    model_id = pretrained_model
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)
elif text_generation_model == "ppo_model":
    model_id = ppo_model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)
else:
    pretrained_model = "facebook/blenderbot-400M-distill"
    model_id = pretrained_model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32).to(device)

# load tokenizer
if finetune or text_generation_model == "blenderbot":
    tokenizer_id = f"facebook/blenderbot-400M-distill"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    emo_labels = ['sadness', 'disappointment', 'neutral', 'fear', 'nervousness', 'disapproval', 'realization', 'annoyance', 'grief', 'approval', 'caring', 'remorse', 'disgust', 'desire', 'love', 'anger', 'embarrassment', 'joy', 'admiration', 'relief', 'surprise', 'optimism', 'confusion', 'curiosity', 'amusement', 'excitement', 'gratitude', 'pride']
    emo_labels = [f"[{i}]" for i in emo_labels]
    special_tokens_dict = {'additional_special_tokens': emo_labels}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    for i in emo_labels:
        tok_id = tokenizer.convert_tokens_to_ids(i)
    model.resize_token_embeddings(len(tokenizer))
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

#tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# load test dataset
with open(f"{load_path_prefix}modeldata/sp_token_ws_empathy_clean_count_top10_score0.4_emo_test_ED_dataset.p", "rb") as f: #ws_empathy_clean_count_top10_score0.4_emo_test_ED_dataset.p
    [test_dataset] = pickle.load(f)
#test_dataset = Dataset.from_dict(test_dataset)[:10]
test_dataset = Dataset.from_dict(test_dataset).shuffle(seed=2023).select(range(10000))
#test_dataset = Dataset.from_dict(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

with open(f"{load_path_prefix}modeldata/real_toxicity_prompts_1000.p", "rb") as f2:
    [toxic_ds] = pickle.load(f2)
neg_dataset = toxic_ds[:negative_sample_size] #.shuffle(seed=2023).select(range(negative_sample_size))

# preprocessing
def tokenize(sample):
    prompt = sample["prompt"] # prompt
    continuation = sample["target"] # utterance

    sample["input_ids"] = tokenizer.encode(prompt, padding='max_length', truncation=True, max_length=128)
    sample["query"] = {"prompt": tokenizer.decode(sample["input_ids"]), "target": continuation}
    return sample

test_dataset = test_dataset.map(tokenize, batched=False)
test_dataset.set_format(type="torch")

# define weights and other variables for evaluation
emp_weight = 1
fluency_weight = 0
#emp_score_label_weight = [-1, 1, 4]
mean_bleu = 0
emp_ratio = []
BLEU_score_list = []
list_generated_texts = []
sent_bert_score = []
neg_sent_bert_score = []
promcse_similarities = []
neg_promcse_similarities = []

# start evaluation
#try:
with open(f'{output_path_prefix}_text_log_emo_probi_score.txt', 'w') as text_log:
    if text_generation_model == "ppo_model":
        text_log.write(f"PPO Model name: {ppo_model} \n")
    counter = 0
    prompts = []
    targets_list = []
    for test_batch in enumerate(test_dataloader):
        # get responses
        test_query = test_batch[1]
        input_texts = test_query["prompt"]
        prompts = prompts + [it.replace("</s>", "").replace("_comma_", ",") for it in input_texts]
        #query_list.append(test_query["query"])
        target = test_query["target"]
        targets_list = targets_list + [t.replace("</s>", "").replace("_comma_", ",") for t in target]
        input_ids = tokenizer(input_texts, return_tensors="pt", padding='max_length', max_length=128, truncation=True).to(device)
        input_ids = input_ids["input_ids"]
        """set do_sample to True for non-deterministic generation"""
        outputs = model.generate(input_ids, do_sample=False, num_beams=10, max_new_tokens=40, use_cache=True)

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = [g.replace("<pad>", "") for g in generated_texts]
        list_generated_texts = list_generated_texts + generated_texts

        # get PromSCE score
        similarities = promcse_model.similarity(generated_texts, target)
        similarities = [similarities[i][i] for i in range(len(generated_texts))]
        promcse_similarities = promcse_similarities + similarities

        # negative sampling as comparison
        neg_sim = promcse_model.similarity(generated_texts, neg_dataset)
        neg_sim = [neg_sim[i][i] for i in range(len(generated_texts))]
        neg_promcse_similarities = neg_promcse_similarities + neg_sim

        input_texts = [it.replace("_comma_", ",") for it in input_texts]

        for t in range(len(input_texts)):
            text_log.write(f"{counter}, {t} Prompt: {input_texts[t]} \n")
            print(f"{counter}, {t} Prompt: {input_texts[t]}")
            text_log.write(f"{counter}, {t} Response: {generated_texts[t]} \n")
            print(f"{counter}, {t} Response: {generated_texts[t]}")
            text_log.write(f"{counter}, {t} Ref: {target[t]} \n\n")
            print(f"{counter}, {t} Ref: {target[t]}\n\n")
        counter += 1

    print("Start calculating bertScore... ")
    corpus_bertscore_results = bertscore.compute(predictions=list_generated_texts, references=targets_list, lang="en")

    # calculate n-gram frequency dict
    token_gen_txts = []
    for s in list_generated_texts:
        token_gen_txts += nltk.word_tokenize(s.replace(".", ""))
    bigramfd = compute_freq(token_gen_txts, 2)
    trigramfd = compute_freq(token_gen_txts, 3)
    quadgramfd = compute_freq(token_gen_txts, 4)

    # get empathy label and score from empathy classifier
    text_log.write(f"{counter} Get empathy labels. \n")
    emp_results = empathy_classifier(list_generated_texts, padding='max_length', truncation=True, max_length=512)
    text_log.write(f"{counter} Obtained results. \n")
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

    # get mean of scores
    corpus_precision, corpus_recall, corpus_f1 = get_bertscore_results(corpus_bertscore_results)
    mean_promcse_sim = get_mean(promcse_similarities)
    neg_mean_promcse_sim = get_mean(neg_promcse_similarities)

    # write results to output text file
    text_log.write("Bigram: \n")
    bi_common = bigramfd.most_common(10)
    for i in range(len(bi_common)):
        text_log.write(f"{str(bi_common[i])}\n")

    text_log.write("\n\nTrigram: \n")
    tri_common = trigramfd.most_common(10)
    for i in range(len(tri_common)):
        text_log.write(f"{str(tri_common[i])}\n")

    text_log.write("\n\nQuadgram: \n")
    quad_common = quadgramfd.most_common(10)
    for i in range(len(quad_common)):
        text_log.write(f"{str(quad_common[i])}\n")

    text_log.write(f"{counter} Empathy Ratio: no empathy {emp_ratio[0]}, weak empathy {emp_ratio[1]}, strong empathy {emp_ratio[2]}.\n")
    text_log.write(f"{counter} Corpus BertScore: precision {corpus_precision}, recall {corpus_recall}, f1 {corpus_f1}.\n")
    text_log.write(f"PromCSE Similarity: {mean_promcse_sim}. \n")
    text_log.write(f"Negative PromCSE Similarity: {neg_mean_promcse_sim}. \n")
text_log.close()

# write results to another separate output text file for easy access
with open(f'{output_path_prefix}_test_score_log_emo_probi_score.txt', 'w') as score_log:
    score_log.write("Bigram: \n")
    bi_common = bigramfd.most_common(10)
    for i in range(len(bi_common)):
        score_log.write(f"{str(bi_common[i])}\n")

    score_log.write("\n\n Trigram: \n")
    tri_common = trigramfd.most_common(10)
    for i in range(len(tri_common)):
        score_log.write(f"{str(tri_common[i])}\n")

    score_log.write("\n\n Quadgram: \n")
    quad_common = quadgramfd.most_common(10)
    for i in range(len(quad_common)):
        score_log.write(f"{str(quad_common[i])}\n")

    score_log.write(f"\nEmpathy Ratio, no empathy: {emp_ratio[0]}, weak empathy {emp_ratio[1]}, strong empathy: {emp_ratio[2]}\n")
    score_log.write(f"Corpus BertScore: precision {corpus_precision}, recall {corpus_recall}, f1 {corpus_f1}.\n")
    score_log.write(f"PromCSE Similarity: {mean_promcse_sim}. \n")
    score_log.write(f"Negative PromCSE Similarity: {neg_mean_promcse_sim}. \n")
    print(f"Empathy Ratio: no empathy {emp_ratio[0]}, weak empathy {emp_ratio[1]}, strong empathy {emp_ratio[2]}\n")
    print(f"Corpus BertScore: precision {corpus_precision}, recall {corpus_recall}, f1 {corpus_f1}.\n")
    print(f"PromCSE Similarity: {mean_promcse_sim}. \n")
    print(f"Negative PromCSE Similarity: {neg_mean_promcse_sim}. \n")
score_log.close()
"""
except Exception as err:
    with open(f'{output_path_prefix}_test_error_log_emo_probi_score.txt', 'w') as err_log:
        err_log.write(f"Unexpected {err=}, {type(err)=}")
    err_log.close()
"""
