import pickle

import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification, \
    RobertaTokenizerFast, RobertaForSequenceClassification
from trl import PPOConfig
from helper import build_train_dataset, build_pad_dataset, padding, build_pretrain_dataset, get_mean, get_emo_counts, \
    get_js_distance, get_FACE_loss
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from torch.optim import Adam
from transformers import get_scheduler
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


config = PPOConfig(
    model_name="facebook/blenderbot-400M-distill",
    learning_rate=5e-5,
)
load_path_prefix = ""
val = False
num_epochs = 10
save_path_prefix = "pretraining_preprocessed_model"
train_dataset_path = "modeldata/sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_dialogue_dataset.p"
dev_dataset_path = "modeldata/ws_empathy_clean_prompt_emo_validation_dialogue_dataset.p"
min_input_length = 30
max_input_length = 100
train_set_size = 2000 # all
dev_set_size = 8
checkpoint = 500
train_batch_size = 4
dev_batch_size = 4
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
#device = torch.device("mps")
weights = torch.tensor([1, 1.5, 1, 0], device=device) #[nll, div, emo, short_len_penalty] # default: 1, 1.5, 1, 0
w = weights
model_save_path = f"{save_path_prefix}2_lr{config.learning_rate}_w{w[0]}-{w[1]}{w[2]}_FACE_mean_emo_probi_{num_epochs}epochs"

# load dataset
dataset = build_pretrain_dataset(config, dataset_path=train_dataset_path, input_min_text_length=min_input_length, input_max_text_length=max_input_length, size=train_set_size)
dev_dataset = build_pretrain_dataset(config, dataset_path=dev_dataset_path, input_min_text_length=min_input_length, input_max_text_length=max_input_length, size=dev_set_size)

# dataloader
train_dataloader = DataLoader(dataset, shuffle=True, batch_size=train_batch_size)
eval_dataloader = DataLoader(dev_dataset, batch_size=dev_batch_size)

# load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# add special tokens
emo_labels = ['sadness', 'disappointment', 'neutral', 'fear', 'nervousness', 'disapproval', 'realization', 'annoyance', 'grief', 'approval', 'caring', 'remorse', 'disgust', 'desire', 'love', 'anger', 'embarrassment', 'joy', 'admiration', 'relief', 'surprise', 'optimism', 'confusion', 'curiosity', 'amusement', 'excitement', 'gratitude', 'pride']
emo_labels = [f"[{i}]" for i in emo_labels]
special_tokens_dict = {'additional_special_tokens': emo_labels}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
for i in emo_labels:
    tok_id = tokenizer.convert_tokens_to_ids(i)
model.resize_token_embeddings(len(tokenizer))
special_ids = tokenizer.all_special_ids
zeros = [0] * len(tokenizer)
token_ids = list(range(0, len(tokenizer)))
token_freq = dict(zip(token_ids, zeros))

reward_model_id = "SamLowe/roberta-base-go_emotions"
emo_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
emo_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, torch_dtype=torch.float32).to(device)
reward_classifier = pipeline('text-classification', model=reward_model_id, tokenizer=reward_model_id, max_length=128, truncation=True, top_k=None) #, device=0

emp_classifier_model = f"{load_path_prefix}models/roberta-empathy-03-06-2023-18_21_58"
empathy_model_id = emp_classifier_model
empathy_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
empathy_model = RobertaForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32).to(device)
# change
empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer, max_length=512, truncation=True, device=0)

# load optimizer
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
# learning rate scheduler
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# track progress
progress_bar = tqdm(range(num_training_steps))
criterion = torch.nn.CrossEntropyLoss()
model.train()

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0, #0.0
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
counter = 0
list_loss = []
list_dev_loss = []
for epoch in range(num_epochs):
    print(f"Start training epoch {epoch}... ")
    for batch in train_dataloader:
        #batch = {k: v.to(device) for k, v in batch.items()}
        query_tensors = batch["input_ids"]
        outputs = []
        target_ids = []
        list_prompt_texts, list_gen_texts = [], []
        num_short_resp = torch.tensor(0, device=device)
        for q in range(len(query_tensors)):
            q1 = query_tensors[q].nonzero()[-1]
            query = query_tensors[q][:(q1+1)].unsqueeze(0).to(device)
            response = model.generate(query, min_new_tokens=4, max_new_tokens=40, use_cache=True, **generation_kwargs)
            outputs.append(response[0])
            for t in response[0]:
                token_freq[int(t)] += 1
            prompt_text = batch["prompt"][q].replace("_comma_", ",")
            list_prompt_texts.append(prompt_text)
            target_emo = batch["target_emo"][q]
            target_emo_ids = torch.tensor([[tokenizer.encode_plus(f"[{target_emo}]", add_special_tokens=True)["input_ids"][0]]], device=device)
            target_emo_ids_resp = torch.concat((torch.tensor(target_emo_ids), response.clone().detach()), 1)
            gen_text = tokenizer.decode(target_emo_ids_resp.squeeze(), skip_special_tokens=True)
            list_gen_texts.append(gen_text)
            if len(gen_text) < 5:
                num_short_resp += 1
            target_resp = batch["target"][q]
            print(f"{counter}, {q} Prompt: {prompt_text}")
            print(f"{counter}, {q} Response: {gen_text}")
            print(f"{counter}, {q} Target: {target_resp} \n")
            t_ids = tokenizer.encode_plus(batch["target"][q], add_special_tokens=True, padding='max_length', truncation=True, max_length=128)["input_ids"]
            target_ids.append(torch.tensor(t_ids, device=device))

        #target_ids = tokenizer.encode(batch["target"])
        outputs = pad_sequence(outputs, batch_first=True)
        target_ids = pad_sequence(target_ids, batch_first=True)
        #loss = criterion(outputs, target_ids)
        #if outputs.size(dim=0) == target_ids.size(dim=0):
        model_output = model(input_ids=outputs, labels=target_ids)
        nll_loss = model_output.loss # nll loss
        loss_logits = model_output.logits
        print(f"NLL loss: {nll_loss}")

        # Calculate FACE, skip sp tokens
        div_loss = get_FACE_loss(token_freq, outputs, special_ids, loss_logits)
        #loss = torch.tensor([nll_loss, div_loss], requires_grad=True, device=device)
        #loss = sum(weights * loss)

        prompt_results = reward_classifier(list_prompt_texts)
        emo_results = reward_classifier(list_gen_texts)
        emo_loss, mean_emo_loss = get_js_distance(prompt_results, emo_results)
        loss = weights[0] * nll_loss + weights[1] * div_loss + weights[2] * mean_emo_loss #+ weights[3] * num_short_resp
        print(f"Weighted loss: {loss}")
        loss_copy = loss.clone().detach().cpu()
        list_loss.append(loss_copy)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if counter % checkpoint == 0:
            print(f"Saving model at counter {counter}... ")
            model.save_pretrained(f"{model_save_path}-{counter}-loss{format(loss, '.5f')}")
        counter += 1
        if val:
            # validation
            for dev_batch in eval_dataloader:
                dev_query_tensors = \
                tokenizer(dev_batch["prompt"], return_tensors="pt", padding=True, max_length=128, truncation=True,
                          add_special_tokens=True)["input_ids"].to(device)
                response_tensors = []
                response_tensors = model.generate(dev_query_tensors, num_beams=3, min_new_tokens=4, max_new_tokens=40,
                                                  **generation_kwargs)
                target = dev_batch["target"]
                target_ids = tokenizer(target, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True,
                                       max_length=128)["input_ids"].to(device)
                generated_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

                dev_prompts = dev_batch["prompt"]
                response_tensors = pad_sequence(response_tensors, batch_first=True)
                target_ids = pad_sequence(target_ids, batch_first=True)
                prompt_results = reward_classifier(dev_prompts)
                emo_results = reward_classifier(generated_texts)  # TODO: add sp tokens to generated output
                list_resp_emo = []
                for e in range(len(emo_results)):
                    resp_emo = emo_results[e][0]['label']
                    list_resp_emo.append(f"[{resp_emo}]")

                resp_emo_token = tokenizer(list_resp_emo, add_special_tokens=True)["input_ids"]
                resp_emo_ids = torch.tensor([[r[0]] for r in resp_emo_token], device=device)

                resp_w_sp_token = []
                for o in range(len(response_tensors)):
                    resp_w_sp_token.append(
                        torch.concat((resp_emo_ids[o].clone().detach(), response_tensors[o].clone().detach()), 0))

                # response_tensors = pad_sequence(response_tensors, batch_first=True)
                # target_ids = pad_sequence(target_ids, batch_first=True)
                resp_w_sp_token = torch.stack(resp_w_sp_token)
                model_output = model(input_ids=resp_w_sp_token, labels=target_ids)
                nll_loss = model_output.loss  # nll loss

                loss_logits = model_output.logits
                div_loss = get_FACE_loss(token_freq, response_tensors, special_ids, loss_logits)
                """
                # empathy loss
                #emp_result = empathy_classifier(list_gen_texts)
                #emp_penalty = 0
                #for r in emp_result:
                #    for l in r:
                #        if l["label"] == 'LABEL_0':
                #            emp_penalty += l['score']
                #norm_emp_penalty = emp_penalty / len(list_gen_texts)
                """

                emo_loss, mean_emo_loss = get_emo_counts(prompt_results, emo_results)
                dev_loss = weights[0] * nll_loss + weights[1] * div_loss + weights[2] * mean_emo_loss  # + weights[3] * num_short_resp
                list_dev_loss.append(dev_loss)
                print(f"Validation loss: {dev_loss}")
                """
                """

model.save_pretrained(f"{model_save_path}-last")
f3 = open(f"{model_save_path}_loss.p", 'wb')
pickle.dump([list_loss, list_dev_loss], f3)
f3.close()