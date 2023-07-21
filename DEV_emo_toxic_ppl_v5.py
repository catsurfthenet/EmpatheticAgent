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
import os
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
    AutoModelForSequenceClassification, pipeline
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
                    load_toxicity_classifier, load_empathy_classifier, load_emo_classifier, emo_dis_bleu, append_scores)
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
input_batch_size = 24
optimiser_choice = "Adam"
reward_function = emo_dis_ppl_toxic
# define weights
emp_weight = 1 #0
toxicity_weight = 0
fluency_weight = 0 #1
ref_gen_emo_match_reward = 2
reward_scale = 1

lr = 1.47e-5
weight_decay = 0.001
ppo_epoch_num = 10
DEV = True
train_set_size = -1 #3000
dev_set_size = 80
checkpoint = 50
epoch_num = 3 # number of outer loops
shared_layers = 4

train_dataset_path = "modeldata/input_resp_clean_prompt_emo_train_dialogue_dataset.p" # 'modeldata/emo_dis_train_dialogue_dataset.p' #"modeldata/ws_empathy_clean_prompt_emo_train_dialogue_dataset.p"  #
save_path_prefix = f"best_cand3_outer{epoch_num}_share{shared_layers}_scale{reward_scale}_inp_resp_clean_ED_emo_resp{ref_gen_emo_match_reward}xppl_{optimiser_choice}_bs{input_batch_size}_lr{lr}_gamma0.99_emo_probi_w1-0-0" #"DEV_lr-7_ppl_toxic_w4-6-0" #"DEV-mimic-lr-6-ppl-toxic" # "DEV_SGD_lr-9_emo_toxic_w6-4-0"
load_path_prefix = "./"
ppo_model = f"{load_path_prefix}DEV_lr-9_ppl_toxic_w6-4-0-blenderbot-400m-emo-probi-ppl-last-score0.6292313380390405-ppl4.034670352935791"


# fixed variables
score_min = 100
score_max = 0
input_mini_batch_size = 4
blenderbot = "facebook/blenderbot-400M-distill"
dialogpt = "microsoft/DialoGPT-medium"
opt_iml = "facebook/opt-iml-1.3b"
model_path = blenderbot

device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
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

def padding(data):
    padded = rnn_utils.pad_sequence(data)
    padded = list(map(torch.Tensor, padded.T))
    return padded

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    steps=10000, # NO: 10k (more neg on all tests)
    learning_rate=script_args.learning_rate,
    adap_kl_ctrl=True,
    #init_kl_coef=0.0001, # NO: 0.05
    #target=0.01,
    #horizon=2048, #
    gamma=0.99,
    #lam=0.95,
    cliprange=0.2,
    cliprange_value=0.2,
    log_with=script_args.log_with,
    ppo_epochs=ppo_epoch_num,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=True,
    compare_steps=1,
)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.

# We retrieve the dataloader by calling the `build_dataset` function.
min_input_length = 30
max_input_length = 100
dataset = build_train_dataset(config, dataset_path=train_dataset_path, input_min_text_length=min_input_length, input_max_text_length=max_input_length, size=2000)
dev_dataset = build_pad_dataset(config, dataset_path='modeldata/dev_dialogue_dataset.p', input_min_text_length=min_input_length, input_max_text_length=max_input_length, size=dev_set_size)
dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer. We first load the model
# in bfloat16 to save memory using `transformers`.
model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, torch_dtype=torch.float32)

# Pass the loaded model to `AutoModelForSeq2SeqLMWithValueHead`.
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model)

#ref_model = create_reference_model(model)

# We make sure to use `Adam` optimizer on the model parameters that require gradients.
#criterion = torch.nn.CrossEntropyLoss()
if optimiser_choice == "AdamW":
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                      weight_decay=weight_decay)
elif optimiser_choice == "SGD":
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
else: # use Adam as default
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# only for this model.
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
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
blenderbot_model = AutoModelForSeq2SeqLM.from_pretrained(blenderbot, torch_dtype=torch.float32).to(ppo_trainer.accelerator.device)
#dev_dataset = dev_dataset.to(ppo_trainer.accelerator.device)


#reward_model_id = "bdotloh/roberta-base-empathy"
reward_model_id = "SamLowe/roberta-base-go_emotions"
empathy_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
empathy_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, torch_dtype=torch.float32).to(
    ppo_trainer.accelerator.device
)
reward_classifier = pipeline('text-classification', model=reward_model_id, tokenizer=reward_model_id, max_length=128, truncation=True, top_k=None)

#if toxicity_weight > 0:
toxicity_model_id = "martin-ha/toxic-comment-model"
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_id)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float32).to(
    ppo_trainer.accelerator.device
)
toxicity_classifier = pipeline('text-classification', model=toxicity_model_id, tokenizer=toxicity_model_id, max_length=128, truncation=True)

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
short_length_penalty = 5
#epoch_num = 0
#init = time.time()
with open(f'{save_path_prefix}_score_train_output.txt', 'w') as f:
    for _ in range(epoch_num):
        for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            #start_loop = time.time()
            query_tensors = batch["input_ids"]
            # Get response from the policy model
            response_tensors = []
            for query in query_tensors:
                gen_len = output_length_sampler()
                generation_kwargs["max_new_tokens"] = gen_len
                response = ppo_trainer.generate(query, **generation_kwargs)
                if len(response) < 4:
                    response = torch.nn.functional.pad(response, (1,2), "constant", 0)
                response_tensors.append(response.squeeze()[-gen_len:])
            #resp_gen_time = time.time()
            #print(f"Generate response in {resp_gen_time - start_loop}")
            response_tensors = padding(response_tensors)
            #response_tensors_squeeze = [r.squeeze() for r in response_tensors]
            batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            #decode_resp = time.time()
            #print(f"Decode resp in {decode_resp - resp_gen_time}")

            texts = batch["response"]
            prompts = [q.get("prompt").replace("</s>", "").replace("_comma_", ",") for q in batch["query"]]
            response_tensors_t = torch.stack(response_tensors)
            # use ref_model, or other models
            loss = blenderbot_model(input_ids=response_tensors_t, labels=response_tensors_t).loss # get loss
            #loss = criterion()
            ppl = float(torch.exp(loss).cpu().detach().numpy())
            mean_ppl_list.append(ppl)
            ppl = min(ppl, 100000) # clip perplexity to within 100,000
            # rescale perplexity to between 1 and 1000
            # the experimental lowest value is around 4
            ppl = 1 if (ppl < 5) else (ppl - 4)
            inverse_ppl = 1 / ppl # inverse perplexity
            #ppl_time = time.time()
            #print(f"Get ppl in {ppl_time - decode_resp}")
            score_list = []

            if counter % checkpoint == 0:
                #start_reg_write = time.time()
                f.write(f"Score min: {score_min}, score max: {score_max}")
                f.write(f"Counter: {counter}, best score: {best_score}")
                f.write('\n')
                for q in range(len(batch["query"])):
                    query = batch["query"][q].get("prompt").replace("</s>", "").replace("_comma_", ",").replace("<pad>", "")
                    response = batch["response"][q]
                    target = batch["query"][q].get("target")
                    f.write(f"{q} Prompt: {query}\n")
                    f.write(f"{q} Response: {response}\n")
                    f.write(f"{q} Ref: {target}\n")
                    f.write('\n')
                #end_reg_write = time.time()
                #print(f"Record % 100 epoch in {end_reg_write - start_reg_write}")

        # Compute sentiment score # noqa
        #start_model_eval = time.time()
        toxicity_results = []
        prompt_results = reward_classifier(prompts)
        emo_results = reward_classifier(texts)
        ref_resp_emo = Dataset.from_list(batch["query"])["target_emo"]
        current_match_reward = 0
        for r in range(len(ref_resp_emo)):
            if prompt_results[r][0]["label"] == ref_resp_emo[r]:
                current_match_reward += (ref_gen_emo_match_reward * inverse_ppl)  #

        if toxicity_weight > 0:
            toxicity_results = toxicity_classifier(texts)
        #end_model_eval = time.time()
        #print(f"3 Model evaluations in {end_model_eval - start_model_eval}")

        print(f"PPL: {ppl}, inverse ppl: {inverse_ppl}")
        #score_list, _, _ = emo_dis_ppl(prompt_results, emo_results, inverse_ppl, weights=[emp_weight, fluency_weight])
        score_list, _, mean_emo_score, mean_toxic_score = emo_dis_ppl_toxic(prompt_results, emo_results, inverse_ppl, toxicity_results, weights=[emp_weight, toxicity_weight, fluency_weight])
        mean_emo_list.append(mean_emo_score)
        mean_toxic_list.append(mean_toxic_score)
        #end_get_score = time.time()
        #print(f"Get score in {end_get_score - end_model_eval}")

        if max(score_list) > score_max:
            score_max = max(score_list)
        if min(score_list) < score_min:
            score_min = min(score_list)

        score_list = [((x - 1e-15) if x == 1.0 else x) for x in score_list]
        print(f"Score min: {score_min}, score max: {score_max} \n")
        #print(f"Score list: {score_list} \n")
        mean_score = get_mean(score_list)
        mean_score_list.append(mean_score)
        for r in range(len(ref_resp_emo)):
            if prompt_results[r][0]["label"] == ref_resp_emo[r]:
                current_match_reward += (ref_gen_emo_match_reward * inverse_ppl)  #
        score_list = logit(score_list) + current_match_reward


        rewards = [torch.tensor(output) for output in score_list] # change reward
        #end_get_result = time.time()
        #print(f"Convert score to logit and tensor in {end_get_result - end_get_score}")

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        list_stats.append(stats)
        ppo_trainer.log_stats(stats, batch, rewards)
        #end_ppo_step = time.time()
        #print(f"PPO steps in {end_ppo_step - end_get_result}")

        if counter % 100 == 0:
            f.write(f"End step {counter}")
            f.write('\n')
        counter += 1

        # Save model every 200 epochs if model has improved in performance
        if counter % checkpoint == 0 and DEV:
            # validate
            try:
                print(f"Start validation for epoch {step} with counter {counter}.")
                BLEU_score_list = []
                prompts = []
                texts = []
                ppl_list = []

                # for dev_query in dev_dataset:
                for dev_batch in enumerate(dev_dataloader):
                    dev_query = dev_batch[1]
                    input_texts = dev_query["prompt"]
                    prompts += input_texts
                    # input_ids = tokenizer(input_texts, return_tensors="pt", padding='max_length', max_length=128,
                    #                      truncation=True).to(ppo_trainer.accelerator.device)
                    input_ids = dev_query["input_ids"].to(ppo_trainer.accelerator.device)  # input_ids["input_ids"]
                    outputs = ppo_trainer.model.generate(input_ids, do_sample=True, num_beams=10, max_new_tokens=40,
                                                         use_cache=True)

                    # print("Generated output")
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
                    dev_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    texts += dev_response
                    """
                    dev_response = word_tokenize(dev_response[0])
                    dev_target = word_tokenize(dev_query["target"])
                    dev_BLEUscore = weighted_bleu_score(dev_target, dev_response)
                    BLEU_score_list.append(dev_BLEUscore)
                    """
                #dev_end_resp_gen = time.time()
                #print(f"Dev response generation time {dev_end_resp_gen - end_ppo_step}")
                #mean_bleu = sum(BLEU_score_list) / len(BLEU_score_list)
                mean_ppl = get_mean(ppl_list)
                # calculate emo distribution
                prompt_results = reward_classifier(prompts)
                emo_results = reward_classifier(texts)
                toxicity_results = toxicity_classifier(texts)
                list_current_score, list_emo_score, mean_emo_score, mean_toxic_score = reward_function(prompt_results, emo_results, inverse_ppl,toxicity_results, weights=[emp_weight, toxicity_weight, fluency_weight])
                #dev_end_score = time.time()
                #print(f"Dev get score in {dev_end_score - dev_end_resp_gen}")

                # current_score = mean((emo_score * emp_weight) + (bleu * fluency_weight))
                current_score = sum(list_current_score) / len(list_current_score)
                print(f"\nSaving model at step {step}. \n")
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(
                        f"{model_save_path}-step{counter}-score{np.float32(current_score)}-ppl{np.float32(mean_ppl)}")
                f.write(f"\nSaving model at step {counter}. \n")
                if current_score > best_score:
                    best_score = current_score
                    f.write(f"Mean perplexity of this step: {mean_ppl}. \n")
                    f.write(f"Mean Emo Score of this step: {mean_emo_score} \n")
                    f.write(f"Score of this step: {current_score}. \n")
                    #dev_save_resp = time.time()
                    #print(f"Dev save response in {dev_save_resp - dev_end_score}")
            except Exception as err:
                with open(f'{save_path_prefix}_error_log_empathy_score_epoch{step}.txt', 'w') as err_log:
                    err_log.write(f"Unexpected {err=}, {type(err)=}")
                err_log.close()
            #end_dev = time.time()
            #print(f"Validation in {end_dev - end_ppo_step}")

    # validate at very last ?
    if DEV:
        try:
            print(f"Start validation for step {step} with counter {counter}.")
            BLEU_score_list = []
            prompts = []
            texts = []
            ppl_list = []
            for dev_batch in enumerate(dev_dataloader):
                # for dev_query in dev_dataset:
                dev_query = dev_batch[1]
                input_texts = dev_query["prompt"]
                prompts += input_texts
                input_ids = dev_query["input_ids"].to(ppo_trainer.accelerator.device)
                # gen_len = output_length_sampler()
                # generation_kwargs["max_new_tokens"] = gen_len
                outputs = ppo_trainer.model.generate(input_ids, do_sample=True, max_new_tokens=40, use_cache=True)
                loss = model(input_ids=outputs, labels=outputs)[1]
                ppl = torch.exp(loss).cpu().detach().numpy()
                ppl_list.append(ppl)
                inverse_ppl = 1 / ppl
                dev_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                texts += dev_response
                """
                dev_response = word_tokenize(dev_response[0])
                dev_target = word_tokenize(dev_query["target"])
                dev_BLEUscore = weighted_bleu_score(dev_target, dev_response)
                BLEU_score_list.append(dev_BLEUscore)
                """

            #mean_bleu = sum(BLEU_score_list) / len(BLEU_score_list)
            print("Start score calculation...")
            toxicity_results = []
            mean_ppl = sum(ppl_list) / len(ppl_list)
            # calculate emo distribution
            prompt_results = reward_classifier(prompts)
            emo_results = reward_classifier(texts)
            if toxicity_weight > 0:
                toxicity_results = toxicity_classifier(texts)
            print("Got results for toxicity ")
            list_current_score, list_emo_score, mean_emo_score, mean_toxic_score = reward_function(prompt_results, emo_results, inverse_ppl, toxicity_results, weights=[emp_weight, toxicity_weight, fluency_weight])
            print("Update min and max score... ")
            if max(list_current_score) > score_max:
                score_max = max(list_current_score)
            if min(list_current_score) < score_min:
                score_min = min(list_current_score)
            print("Write min and max score... ")
            f.write(f"Score min: {score_min}, score max: {score_max}")
            f.write(f"Score list: {list_current_score}")
            """
            list_emo_score, mean_emo_score = get_js_distance(prompt_results, emo_results)
            BLEU_score_list = [(b) * fluency_weight for b in BLEU_score_list]
            list_emo_score = [e * emp_weight for e in list_emo_score]
            current_score = [sum(x) for x in zip(BLEU_score_list, list_emo_score)]
            mean_score = sum(current_score) / len(current_score)
            """
            current_score = sum(list_current_score) / len(list_current_score)

            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(f"{model_save_path}-last-score{current_score}-ppl{np.float32(mean_ppl)}")
        except Exception as err:
            with open(f'{save_path_prefix}_error_log_emo_probi_score_last.txt', 'w') as err_log:
                err_log.write(f"Unexpected {err=}, {type(err)=}")
            err_log.close()
    f3 = open(f"{save_path_prefix}_mean_score_list.p", 'wb')
    pickle.dump([mean_score_list, mean_emo_list, mean_toxic_list, mean_ppl_list, list_stats], f3)
    f3.close()
f.close()