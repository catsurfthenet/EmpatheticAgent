# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import os, psutil
import pickle
from datasets import Dataset, DatasetDict
from dataclasses import dataclass, field
from typing import Optional
import time
import torch
from datasets import load_dataset
from torch.optim import Adam, SGD, AdamW
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification, pipeline, RobertaTokenizerFast, BertTokenizer, BertModel
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed, \
    AutoModelForSeq2SeqLMWithValueHead
from trl.core import LengthSampler
from scipy.special import logit
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
#from classifiers import get_sentence_score
from scipy.spatial import distance
from scipy.special import softmax
from helper import (build_dataset, build_pad_dataset, build_train_dataset, get_mean,
                    weighted_bleu_score, get_js_distance, emo_dis_ppl, emo_dis_ppl_toxic,
                    load_toxicity_classifier, load_empathy_classifier, load_emo_classifier, emo_dis_bleu, append_scores,
                    emo_count_ppl_toxic, get_empathy_ratio, build_dataset_no_token, get_FACE_reward, get_FACE_loss,
                    get_FACE_loss_mod)
#from torch.utils.data import DataLoader
tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPTJ model to generate less toxic contents
# by using allenai/real-toxicity-prompts dataset. We use PPO
#  (proximal policy optimization) to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################
"""ED: empathetic dialogues, RT: real toxic"""
# define path and variables
SAVE_MODEL = False
input_batch_size = 16 #24
optimiser_choice = "Adam"
reward_function = emo_count_ppl_toxic

# define weights
toxicity_weight = 0
fluency_weight = 0.3 #1 increase this?
emp_weight = 1.25 #0
div_weight = 1.25 #0
sim_weight = 1.25
w = [emp_weight, div_weight, sim_weight, fluency_weight]
ref_gen_emo_match_reward = 0
reward_scale = 1
use_target_steps = 0
optim_maximise = False

lr = 1e-8#1e-5 #-9
weight_decay = 0.001
ppo_epoch_num = 4 #4
DEV = False
train_set_size = 2000 # -1 #3000
dev_set_size = 100
checkpoint = 50
epoch_num = 1 #1 # number of outer loops
shared_layers = 9 #4
gamma = 1 #0.90
episode_num = 2000 # 500~1000

date = datetime.datetime.now()
train_dataset_path = "modeldata/ws_empathy_clean_prompt_emo_train_dialogue_dataset.p" #'modeldata/emo_count_train_dialogue_dataset.p' #"modeldata/ws_empathy_clean_prompt_emo_train_dialogue_dataset.p"  #
save_path_prefix = f"PPO{date.day}-{date.month}_eg_tKL1.5_ep{episode_num}_ED_ts{train_set_size}_outer{epoch_num}_inner{ppo_epoch_num}_share{shared_layers}_scale{reward_scale}_empathy_clean_count0.4_pnll{w[3]}_emoCount{w[0]}_FACE{w[1]}_bertSim{w[2]}_{optimiser_choice}_bs{input_batch_size}_lr{lr}_gamma{gamma}" #"DEV_lr-7_ppl_toxic_w4-6-0" #"DEV-mimic-lr-6-ppl-toxic" # "DEV_SGD_lr-9_emo_toxic_w6-4-0"
load_path_prefix = ""
ppo_model = f"{load_path_prefix}w0.5-2.0-1.0-1.0-1.0_best_50epochs-3750-loss1.01517"


# fixed variables
score_min = 100
score_max = 0
input_mini_batch_size = 4
blenderbot = f"{load_path_prefix}models/local-facebook-blenderbot-400M-distill"
dialogpt = "microsoft/DialoGPT-medium"
opt_iml = "facebook/opt-iml-1.3b"
model_path = blenderbot #ppo_model#

device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
#device = torch.device("mps")

# the below section is modified from a huggingface PPO usage example
# https://github.com/huggingface/trl/blob/main/examples/research_projects/toxicity/scripts/gpt-j-6b-toxicity.py
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default=model_path, metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(lr) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=input_mini_batch_size, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=input_batch_size, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default=f"./{save_path_prefix}", # blenderbot-400M-distill-empathy-score-only
        metadata={"help": "the path to save the model"},
    )

"""
Padding function, pad tensors so that all tensors in list have same dimension

Parameters:
data: list of tensors
    input to be padded
    
Return 
padded: list of tensors
    padded list of tensors
"""
def padding(data):
    padded = rnn_utils.pad_sequence(data)
    padded = list(map(torch.Tensor, padded.T))
    return padded

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# configurations of PPO
config = PPOConfig(
    model_name=script_args.model_name,
    steps=10000, # NO: 10k (more neg on all tests)
    learning_rate=script_args.learning_rate, #5e-6,#
    adap_kl_ctrl=True,
    init_kl_coef=0.2, # NO: 0.05
    target=1.5, #3
    horizon=10000, #
    gamma=1,#gamma,
    #lam=0.95,
    cliprange=0.2, #.2
    cliprange_value=0.2, #.2
    vf_coef=.1,
    log_with=script_args.log_with,
    ppo_epochs=4,#ppo_epoch_num,
    mini_batch_size=4, #script_args.mini_batch_size,
    batch_size=16, #script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=True,
    compare_steps=1,
    seed=43,
)
"""
config = {
        "steps": 10000,
        "batch_size": 16, # 2
        "forward_batch_size": 4, # 2
        "ppo_epochs": 4,
        "max_len": 50,
        "lr": 5e-6,
        "init_kl_coef":0.2,
        "seed": 1,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
}
"""

min_input_length = 30
max_input_length = 100

# setup data loader using the `build_dataset_no_token` modified from the `build_dataset` function
dataset = build_dataset_no_token(config, dataset_path=train_dataset_path, input_min_text_length=min_input_length, input_max_text_length=max_input_length, size=train_set_size)
dev_dataset = build_dataset_no_token(config, dataset_path='modeldata/ws_empathy_clean_prompt_emo_validation_dialogue_dataset.p', input_min_text_length=min_input_length, input_max_text_length=max_input_length, size=dev_set_size)

"""
print("LOADING empathetic_dialogue")
with open(train_dataset_path, "rb") as f:
    [data] = pickle.load(f)
dataset = Dataset.from_dict(data)[:train_set_size]
dataset["input_ids"] = dataset["prompt"]
dataset["query"] = dataset["target"]
#dataset["query"] = {"prompt": dataset["prompt"],
#                    "target": dataset["target"],
#                    "prompt_emo": dataset["prompt_emo"],
#                    "target_emo": dataset["target_emo"]}
dataset = Dataset.from_dict(dataset)
f.close()
dev_dataset_path = 'modeldata/dev_dialogue_dataset.p'
with open(dev_dataset_path, "rb") as fd:
    [dev_data] = pickle.load(fd)
dev_dataset = Dataset.from_dict(dev_data)[:dev_set_size]
dev_dataset = Dataset.from_dict(dev_dataset)
fd.close()
"""
#train_dataloader = DataLoader(dataset, batch_size=input_batch_size, shuffle=False)
# data loader for validation dataset
dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# load pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, torch_dtype=torch.float32)

# setup optimiser according to the specified choice
if optimiser_choice == "AdamW":
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                      weight_decay=weight_decay)
elif optimiser_choice == "SGD":
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
else: # use Adam as default
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, maximize=optim_maximise)

# load tokeniser depending on which pretained model is used
if model_path == ppo_model:
    tokenizer = AutoTokenizer.from_pretrained(blenderbot)
else:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# add special tokens
emo_labels = ['sadness', 'disappointment', 'neutral', 'fear', 'nervousness', 'disapproval', 'realization', 'annoyance', 'grief', 'approval', 'caring', 'remorse', 'disgust', 'desire', 'love', 'anger', 'embarrassment', 'joy', 'admiration', 'relief', 'surprise', 'optimism', 'confusion', 'curiosity', 'amusement', 'excitement', 'gratitude', 'pride']
emo_labels = [f"[{i}]" for i in emo_labels]
special_tokens_dict = {'additional_special_tokens': emo_labels}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
for i in emo_labels:
    tok_id = tokenizer.convert_tokens_to_ids(i)
model.resize_token_embeddings(len(tokenizer))

# setup token frequency
special_ids = tokenizer.all_special_ids
zeros = [0] * len(tokenizer)
token_ids = list(range(0, len(tokenizer)))
token_freq = dict(zip(token_ids, zeros))
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Pass the loaded model to `AutoModelForSeq2SeqLMWithValueHead`.
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model)

# build PPOTrainer, create reference model by specifying the number of shared layers defined above
ppo_trainer = PPOTrainer(
    config,
    model,
    #ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
    num_shared_layers=shared_layers, # total number of layers 11
)

# We then build the reward pipeline, we will use the emotion classification model to compute the reward.
model = model.to(ppo_trainer.accelerator.device)
blenderbot_model = AutoModelForSeq2SeqLM.from_pretrained(ppo_model, torch_dtype=torch.float32).to(ppo_trainer.accelerator.device)
blenderbot_model.resize_token_embeddings(len(tokenizer))
#dev_dataset = dev_dataset.to(ppo_trainer.accelerator.device)


#reward_model_id = "bdotloh/roberta-base-empathy"
reward_model_id = f"{load_path_prefix}models/local-SamLowe-roberta-base-go_emotions"
#emo_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
#emo_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, torch_dtype=torch.float32).to(
#    ppo_trainer.accelerator.device
#)
reward_classifier = pipeline('text-classification', model=reward_model_id, tokenizer=reward_model_id, max_length=128, truncation=True, top_k=None, device=0) #, device=0

#if toxicity_weight > 0:
emp_classifier_model = f"{load_path_prefix}models/roberta-empathy-03-06-2023-18_21_58"
empathy_model_id = emp_classifier_model
empathy_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
empathy_model = RobertaForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32).to(
    device)
# change
empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer,
                              max_length=512, truncation=True, device=0) #

bert_model = BertModel.from_pretrained(f"{load_path_prefix}models/local-bert-base-uncased").to(device)
bert_tokenizer = BertTokenizer.from_pretrained(f"{load_path_prefix}models/local-bert-base-uncased")
cos = torch.nn.CosineSimilarity() #dim=1

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0, #0.0
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 20 #20
output_max_length = 50
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = script_args.model_save_path
counter = 0
best_score = 0
mean_score_list = []
mean_ppl_list = []
mean_emo_list = []
mean_toxic_list = []
list_stats = []
list_emp_ratio = []
short_length_penalty = 5
input_size = LengthSampler(min_input_length, max_input_length)
#epoch_num = 0
#init = time.time()
with open(f'{save_path_prefix}_score_train_output.txt', 'w') as f:
    for _ in range(epoch_num):
        # for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        for ep in range(episode_num):
            torch.cuda.empty_cache()
            ep_data = dataset.shuffle(seed=ep).select(range(16))
            batch = ep_data
            query_tensors = tokenizer(batch["input_ids"], padding=True, max_length=128, truncation=True, add_special_tokens=True)["input_ids"]
            # Get response from the policy model
            query_tensors = [torch.tensor(q, device=device) for q in query_tensors] # need list of tensors
            response_tensors = ppo_trainer.generate(query_tensors, num_beams=3, min_new_tokens=10, max_new_tokens=100, **generation_kwargs)

            # record output token frequency to include new tokens
            for r in response_tensors:
                for token in r:
                    token_freq[int(token)] += 1

            # decode response tokens
            og_response_tensors = response_tensors
            response_tensors = padding(response_tensors)
            #batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

            #texts = batch["response"]
            texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            prompts = [q.replace("</s>", "").replace("_comma_", ",") for q in batch["input_ids"]]
            response_tensors_t = torch.stack(response_tensors).to(device)

            # get target token ids
            target_txt = [q.get("target") for q in batch["query"]]
            target_ids = tokenizer(target_txt, return_tensors="pt", padding=True, max_length=128, truncation=True, add_special_tokens=True)["input_ids"].to(device)

            # use ref_model, or other models to calculate perplexity
            loss = blenderbot_model(input_ids=response_tensors_t, labels=response_tensors_t).loss # get loss
            model_output = model(input_ids=response_tensors_t, labels=target_ids)
            nll_loss = model_output[1].cpu().detach().numpy() # (logit, loss, ?)
            loss_logits = model_output[0].cpu().detach().numpy()
            ppl = float(torch.exp(loss).cpu().detach().numpy())
            mean_ppl_list.append(ppl)
            #ppl = min(ppl, 1000) # clip perplexity to within 1,000
            # rescale perplexity to between 1 and 1000
            # the experimental lowest value is around 4
            #ppl = 1 if (ppl < 5) else (ppl - 4)

            # calculate inverse perplexity
            inverse_ppl = 1 / ppl
            print(f"PPL: {ppl}, inverse ppl: {inverse_ppl}")

            score_list = []
            toxicity_results = []

            # Compute emotion rewards and semantic rewards
            # get emotion labels
            prompt_results = reward_classifier(prompts)
            emo_results = reward_classifier(texts)

            # calculate BERT embedding similarity reward
            ref_resp_emo = Dataset.from_list(batch["query"])["target_emo"]
            target_w_sp = Dataset.from_list(batch["query"])["target"]
            target_ids = tokenizer(target_w_sp, return_tensors="pt", padding=True, max_length=128, truncation=True, add_special_tokens=True)["input_ids"].to(device)
            target_txt = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
            bert_output_token = bert_tokenizer(texts, return_tensors="pt", padding='max_length', max_length=64, truncation=True)["input_ids"].to(device)
            bert_target_token = bert_tokenizer(target_txt, return_tensors="pt", padding='max_length', max_length=64, truncation=True)["input_ids"].to(device)
            output_embeddings = bert_model.embeddings.word_embeddings(bert_output_token)
            target_embeddings = bert_model.embeddings.word_embeddings(bert_target_token)
            sim_reward = 0
            for e in range(len(output_embeddings)):
                sim_reward += cos(output_embeddings[e], target_embeddings[e])
            norm_sim_reward = float(sum(sim_reward) / (len(sim_reward) * len(texts)))
            current_match_reward = 0
            #score_list, _, _ = emo_dis_ppl(prompt_results, emo_results, inverse_ppl, weights=[emp_weight, fluency_weight])

            # calculate emotion reward, save in score_list
            score_list, _, mean_emo_score, mean_toxic_score = reward_function(prompt_results, emo_results, inverse_ppl, toxicity_results, weights=[emp_weight, toxicity_weight, fluency_weight])
            mean_emo_list.append(mean_emo_score)
            mean_toxic_list.append(mean_toxic_score)

            # calculate diversity reward
            div_reward = get_FACE_reward(token_freq=token_freq, outputs=response_tensors_t, special_ids=special_ids)
            #div_loss = get_FACE_loss_mod(token_freq, response_tensors, special_ids, loss_logits)

            #model_output = model(input_ids=response_tensors_t, labels=target_ids)
            #nll_loss = model_output[1]  #logits # nll loss

            # calculate weighted score
            score_list = fluency_weight * (-nll_loss) + emp_weight * np.array(score_list) + div_weight * (div_reward) + norm_sim_reward * sim_weight #- 0.5 * float(nll_loss)

            # record weighted score
            mean_score = get_mean(score_list)
            mean_score_list.append(mean_score)
            rewards = [torch.tensor(output) for output in score_list]

            if max(score_list) > score_max:
                score_max = max(score_list)
            if min(score_list) < score_min:
                score_min = min(score_list)
            print(f"Score min: {score_min}, score max: {score_max} \n")

            if counter % checkpoint == 0:
                f.write(f"Score min: {score_min}, score max: {score_max}")
                f.write(f"Episode: {ep}, best score: {best_score} \n")
                for q in range(len(batch["query"])):
                    query = batch["query"][q].get("prompt").replace("</s>", "").replace("_comma_", ",").replace("<pad>", "")
                    response = texts[q]
                    target = batch["query"][q].get("target")
                    f.write(f"{q} Prompt: {query}\n")
                    f.write(f"{q} Response: {response}\n")
                    f.write(f"{q} Ref: {target}\n")
                    f.write(f"End step {counter} \n \n")

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, og_response_tensors, rewards)
            list_stats.append(stats)
            #ppo_trainer.log_stats(stats, batch, rewards)

            # get empathy ratio for record use, not used for rewards
            emp_results = empathy_classifier(texts, padding='max_length', truncation=True, max_length=512)
            emp_ratio = get_empathy_ratio(emp_results)
            list_emp_ratio.append(emp_ratio)
            f.write(
                f"{ep} Empathy Ratio: no empathy {emp_ratio[0]}, weak empathy {emp_ratio[1]}, strong empathy {emp_ratio[2]}.\n")
            print(
                f"{ep} Empathy Ratio: no empathy {emp_ratio[0]}, weak empathy {emp_ratio[1]}, strong empathy {emp_ratio[2]}.\n")

            counter += 1
            # Save model every checkpoint
            if ep % checkpoint == 0 and SAVE_MODEL:
                print(f"\nSaving model at episode {ep}. \n")
                if ppo_trainer.accelerator.is_main_process:
                    round_ratio = [format(ratio, '.3f') for ratio in emp_ratio]
                    ppo_trainer.save_pretrained(
                        f"{model_save_path}-ep{ep}-ratio{round_ratio[0]}-{round_ratio[1]}-{round_ratio[2]}-ppl{format(ppl, '.3f')}")
                f.write(f"\nSaving model at episode {ep}. \n")

            if ep % checkpoint == 0 and DEV:
                # validate
                try:
                    print(f"Start validation for epoch {counter} with counter {counter}.")
                    BLEU_score_list = []
                    prompts = []
                    texts = []
                    ppl_list = []
                    dev_response_tensors = []

                    for dev_batch in enumerate(dev_dataloader):
                        dev_query = dev_batch[1]
                        input_texts = dev_query["prompt"]
                        prompts += input_texts
                        # input_ids = tokenizer(input_texts, return_tensors="pt", padding='max_length', max_length=128,
                        #                      truncation=True).to(ppo_trainer.accelerator.device)
                        input_ids = dev_query["input_ids"].to(ppo_trainer.accelerator.device)
                        for i in range(len(input_ids)):
                            q1 = input_ids[i].nonzero()[-1]
                            input_ids[i] = input_ids[i][:(q1 + 1)].unsqueeze(0)

                        # generate response
                        outputs = ppo_trainer.model.generate(input_ids, do_sample=True, num_beams=10, max_new_tokens=40,
                                                             use_cache=True)

                        """
                        dev_input_ids = dev_query["input_ids"].to(device)#to(ppo_trainer.accelerator.device)
                        input_shape = dev_input_ids.size()
                        #dev_input_ids = tokenizer(input_texts, return_tensors="pt", padding='max_length', max_length=128, truncation=True).to(device)
                        gen_len = output_length_sampler()
                        generation_kwargs["max_new_tokens"] = gen_len
                        outputs = ppo_trainer.generate(dev_input_ids, do_sample=True, num_beams=10, max_new_tokens=40, use_cache=True)
                        """
                        loss = model(input_ids=outputs, labels=outputs)[1]
                        ppl = torch.exp(loss).cpu().detach().numpy()
                        ppl_list.append(ppl)
                        inverse_ppl = 1 / ppl

                        dev_response_tensors.append(outputs)
                        # decode response
                        dev_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        texts += dev_response
                        """
                        dev_response = word_tokenize(dev_response[0])
                        dev_target = word_tokenize(dev_query["target"])
                        dev_BLEUscore = weighted_bleu_score(dev_target, dev_response)
                        BLEU_score_list.append(dev_BLEUscore)
                        """

                    # get mean of inverse perplexity
                    mean_ppl = get_mean(ppl_list)

                    # get emotion labels and calculate emotion rewards
                    prompt_results = reward_classifier(prompts)
                    emo_results = reward_classifier(texts)
                    toxicity_results = []
                    list_current_score, list_emo_score, mean_emo_score, mean_toxic_score = reward_function(prompt_results, emo_results, mean_ppl, toxicity_results, weights=[emp_weight, toxicity_weight, fluency_weight])

                    # calculate BERT embedding similarity reward
                    ref_resp_emo = Dataset.from_list(dev_batch["query"])["target_emo"]
                    target_w_sp = Dataset.from_list(dev_batch["query"])["target"]
                    target_ids = \
                    tokenizer(target_w_sp, return_tensors="pt", padding=True, max_length=128, truncation=True,
                              add_special_tokens=True)["input_ids"].to(device)
                    target_txt = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
                    bert_output_token = \
                    bert_tokenizer(texts, return_tensors="pt", padding='max_length', max_length=64, truncation=True)[
                        "input_ids"].to(device)
                    bert_target_token = \
                    bert_tokenizer(target_txt, return_tensors="pt", padding='max_length', max_length=64,
                                   truncation=True)["input_ids"].to(device)
                    output_embeddings = bert_model.embeddings.word_embeddings(bert_output_token)
                    target_embeddings = bert_model.embeddings.word_embeddings(bert_target_token)
                    sim_reward = 0
                    for e in range(len(output_embeddings)):
                        sim_reward += cos(output_embeddings[e], target_embeddings[e])
                    norm_sim_reward = float(sum(sim_reward) / (len(sim_reward) * len(texts)))

                    # calculate diversity reward
                    dev_response_tensors = padding(dev_response_tensors)
                    dev_response_tensors = torch.stack(dev_response_tensors).to(device)
                    div_reward = get_FACE_reward(token_freq=token_freq, outputs=dev_response_tensors,
                                                 special_ids=special_ids)

                    # calculate weighted score
                    list_current_score = fluency_weight * mean_ppl + emp_weight * list_current_score + div_weight * div_reward + sim_weight * norm_sim_reward
                    current_score = sum(list_current_score) / len(list_current_score)
                    if current_score > best_score:
                        best_score = current_score
                        f.write(f"Mean perplexity of this step: {mean_ppl}. \n")
                        f.write(f"Mean Emo Score of this step: {mean_emo_score} \n")
                        f.write(f"Score of this step: {current_score}. \n")
                except Exception as err:
                    with open(f'{save_path_prefix}_error_log_empathy_score_epoch{counter}.txt', 'w') as err_log:
                        err_log.write(f"Unexpected {err=}, {type(err)=}")
                    err_log.close()


    # Save at last
    emp_results = empathy_classifier(texts, padding='max_length', truncation=True, max_length=512)
    emp_ratio = get_empathy_ratio(emp_results)
    list_emp_ratio.append(emp_ratio)
    f.write(
        f"{counter} Empathy Ratio: no empathy {emp_ratio[0]}, weak empathy {emp_ratio[1]}, strong empathy {emp_ratio[2]}.\n")
    print(
        f"{counter} Empathy Ratio: no empathy {emp_ratio[0]}, weak empathy {emp_ratio[1]}, strong empathy {emp_ratio[2]}.\n")

    print(f"\nSaving model at counter {counter}. \n")
    if ppo_trainer.accelerator.is_main_process:
        round_ratio = [format(ratio, '.3f') for ratio in emp_ratio]
        ppo_trainer.save_pretrained(
            f"{model_save_path}-last-ratio{round_ratio[0]}-{round_ratio[1]}-{round_ratio[2]}-ppl{format(ppl, '.3f')}")
    f.write(f"\nSaving model at step {counter}. \n")

    # save training data
    f3 = open(f"{save_path_prefix}_mean_score_list.p", 'wb')
    pickle.dump([mean_score_list, mean_emo_list, mean_toxic_list, mean_ppl_list, list_stats, list_emp_ratio], f3)
    f3.close()
f.close()
