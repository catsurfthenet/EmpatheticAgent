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

import torch
from datasets import load_dataset
from torch.optim import Adam
import torch.nn.utils.rnn as rnn_utils
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
from helper import weighted_bleu_score, get_js_distance, emo_dis_bleu, append_scores

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

# define path
save_path_prefix = "DEV-mimic-lr-6-w0.05-0.95"
load_path_prefix = "../"
# define weights
emp_weight = 0.05
fluency_weight = 0.95

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
    model_name: Optional[str] = field(default="facebook/blenderbot-400M-distill", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-7) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default=f"./{save_path_prefix}-blenderbot-400m-emo-probi-bleu", # blenderbot-400M-distill-empathy-score-only
        metadata={"help": "the path to save the model"},
    )

if(os.path.exists('modeldata/emo_probi.p')):
    print("LOADING data emotion probability distribution...")
    with open('modeldata/emo_probi.p', "rb") as f:
        [all_emo_probi, _] = pickle.load(f)
    f.close()
all_emo_probi = dict(all_emo_probi)

def padding(data):
    padded = rnn_utils.pad_sequence(data)
    padded = list(map(torch.Tensor, padded.T))
    return padded

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=1,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config, dataset_path='modeldata/dialogue_dataset.p', input_min_text_length=5, input_max_text_length=100
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

    #ds = load_dataset(dataset_name, split="train")

    if (os.path.exists(dataset_path)):
        print("LOADING empathetic_dialogue")
        with open(dataset_path, "rb") as f:
            [data] = pickle.load(f)
    ds = Dataset.from_dict(data)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"] # prompt
        continuation = sample["target"] # utterance

        sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
        #sample["target_ids"] = tokenizer.encode(continuation)[: input_size()]
        sample["query"] = {"prompt": tokenizer.decode(sample["input_ids"]), "target": continuation}
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
min_input_length = 30
max_input_length = 100
dataset = build_dataset(config, dataset_path='modeldata/dialogue_dataset.p', input_min_text_length=min_input_length, input_max_text_length=max_input_length)
dev_dataset = build_dataset(config, dataset_path='modeldata/dev_dialogue_dataset.p', input_min_text_length=min_input_length, input_max_text_length=max_input_length)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer. We first load the model
# in bfloat16 to save memory using `transformers`.
#model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float32)
model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, torch_dtype=torch.float32)

# And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
#model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model)

# We create a reference model by sharing 20 layers
#ref_model = create_reference_model(model, num_shared_layers=20)
ref_model = create_reference_model(model)

# We make sure to use `Adam` optimizer on the model parameters that require gradients.
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)
"""
# We then build the reward pipeline, we will use the emotion classification model to compute the reward.
# We first load the toxicity model and tokenizer.
emo_class_model_id = "j-hartmann/emotion-english-distilroberta-base" #TODO change this
emo_tokenizer = RobertaTokenizer.from_pretrained(emo_class_model_id)
# We load the toxicity model in fp16 to save memory.
emo_model = RobertaForSequenceClassification.from_pretrained(emo_class_model_id, torch_dtype=torch.float32).to(
    ppo_trainer.accelerator.device
)
"""
model = model.to(ppo_trainer.accelerator.device)
#dev_dataset = dev_dataset.to(ppo_trainer.accelerator.device)

#reward_model_id = "bdotloh/roberta-base-empathy"
reward_model_id = "SamLowe/roberta-base-go_emotions"
empathy_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
empathy_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, torch_dtype=torch.float32).to(
    ppo_trainer.accelerator.device
)
#reward_classifier = pipeline('text-classification', model = reward_model_id)
reward_classifier = pipeline('text-classification', model=reward_model_id, tokenizer=reward_model_id, max_length=512, truncation=True, top_k=None)

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


def reward_function():
    pass


with open(f'{save_path_prefix}_emo_probi_score_train_output.txt', 'w') as f:
    counter = 0
    best_score = 0
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
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
        response_tensors = padding(response_tensors)
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        texts = batch["response"]
        prompts = [q.get("prompt").replace("</s>", "").replace("_comma_", ",") for q in batch["query"]]

        score_list = []

        if counter % 100 == 0:
            f.write(f"Counter: {counter}, best BLEU: {best_score}")
            f.write('\n')
            for q in range(len(batch["query"])):
                query = batch["query"][q].get("prompt")
                response = batch["response"][q]
                target = batch["query"][q].get("target")
                f.write(f"{q} Prompt: {query}\n")
                f.write(f"{q} Response: {response}\n")
                f.write(f"{q} Ref: {target}\n")
                f.write('\n')

        # Compute sentiment score # noqa
        prompt_results = reward_classifier(prompts)
        emo_results = reward_classifier(texts)

        score_list, _, _ = logit(emo_dis_bleu(batch["query"], batch["response"], prompt_results, emo_results, weights=[emp_weight, fluency_weight]))
        rewards = [torch.tensor(output) for output in score_list] # change reward

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if counter % 100 == 0:
            f.write(f"End step {counter}")
            f.write('\n')
        counter += 1

        # Save model every 100 epochs
        if epoch % 200 == 0:
            # validate
            try:
                print(f"Start validation for epoch {epoch} with counter {counter}.")
                BLEU_score_list = []
                prompts = []
                texts = []
                for dev_query in dev_dataset:
                    input_texts = dev_query["prompt"]
                    prompts.append(input_texts)
                    dev_input_ids = dev_query["input_ids"].to(ppo_trainer.accelerator.device)
                    gen_len = output_length_sampler()
                    generation_kwargs["max_new_tokens"] = gen_len
                    outputs = ppo_trainer.generate(dev_input_ids, do_sample=True, max_new_tokens=40, use_cache=True)
                    dev_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    texts.append(dev_response[0])
                    dev_response = word_tokenize(dev_response[0])
                    dev_target = word_tokenize(dev_query["target"])
                    dev_BLEUscore = weighted_bleu_score(dev_target, dev_response)
                    BLEU_score_list.append(dev_BLEUscore)

                mean_bleu = sum(BLEU_score_list) / len(BLEU_score_list)
                # calculate emo distribution
                prompt_results = reward_classifier(prompts)
                emo_results = reward_classifier(texts)
                list_emo_score, mean_emo_score = get_js_distance(prompt_results, emo_results)
                BLEU_score_list = [(b) * fluency_weight for b in BLEU_score_list]
                list_emo_score = [e * emp_weight for e in list_emo_score]
                list_current_score = [sum(x) for x in zip(BLEU_score_list, list_emo_score)]
                # current_score = mean((emo_score * emp_weight) + (bleu * fluency_weight))
                current_score = sum(list_current_score) / len(list_current_score)

                if current_score > best_score:
                    best_score = current_score
                    print(f"\nSaving model at epoch {epoch}. \n")
                    if ppo_trainer.accelerator.is_main_process:
                        ppo_trainer.save_pretrained(f"{model_save_path}-epoch{epoch}-score{np.float32(current_score)}-bleu{np.float32(mean_bleu)}")
                    f.write(f"\nSaving model at epoch {epoch}. \n")
                    f.write(f"Mean BLEU of this epoch: {mean_bleu}. \n")
                    f.write(f"Mean JS distance of this epoch: {1 - mean_emo_score} \n")
                    f.write(f"Score of this epoch: {current_score}. \n")
            except Exception as err:
                with open(f'{save_path_prefix}_error_log_empathy_score_epoch{epoch}.txt', 'w') as err_log:
                    err_log.write(f"Unexpected {err=}, {type(err)=}")
                err_log.close()


    # validate at very last ?
    try:
        print(f"Start validation for epoch {epoch} with counter {counter}.")
        BLEU_score_list = []
        prompts = []
        texts = []
        for dev_query in dev_dataset:
            input_texts = dev_query["prompt"]
            prompts.append(input_texts)
            dev_input_ids = dev_query["input_ids"].to(ppo_trainer.accelerator.device)
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            outputs = ppo_trainer.generate(dev_input_ids, do_sample=True, max_new_tokens=40, use_cache=True)
            dev_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            texts.append(dev_response[0])
            dev_response = word_tokenize(dev_response[0])
            dev_target = word_tokenize(dev_query["target"])
            dev_BLEUscore = weighted_bleu_score(dev_target, dev_response)
            BLEU_score_list.append(dev_BLEUscore)

        mean_bleu = sum(BLEU_score_list) / len(BLEU_score_list)
        # calculate emo distribution
        prompt_results = reward_classifier(prompts)
        emo_results = reward_classifier(texts)
        list_emo_score, mean_emo_score = get_js_distance(prompt_results, emo_results)
        BLEU_score_list = [(1-b) * fluency_weight for b in BLEU_score_list]
        list_emo_score = [e * emp_weight for e in list_emo_score]
        current_score = [sum(x) for x in zip(BLEU_score_list, list_emo_score)]
        mean_score = sum(current_score) / len(current_score)
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(f"{model_save_path}-last-score{mean_score}-bleu{np.float32(mean_bleu)}")
    except Exception as err:
        with open(f'{save_path_prefix}_error_log_emo_probi_score_last.txt', 'w') as err_log:
            err_log.write(f"Unexpected {err=}, {type(err)=}")
        err_log.close()

f.close()
