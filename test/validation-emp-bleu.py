import os
import pickle
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, \
    pipeline
import nltk

device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

context_length = 200

pretrained_model = "facebook/blenderbot-400M-distill"
ppo_model = "./models/blenderbot-400m-empathy-bleu-test1epoch-sarcasm"
model_id = ppo_model
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def tokenize(sample):
    prompt = sample["prompt"]  # prompt
    continuation = sample["target"]  # utterance

    sample["input_ids"] = tokenizer.encode(prompt)
    sample["query"] = {"prompt": tokenizer.decode(sample["input_ids"]), "target": continuation}
    return sample


if (os.path.exists('modeldata/dev_concat_dataset.p')):
    print("LOADING concatenated dev empathetic_dialogue")
    with open('modeldata/dev_concat_dataset.p', "rb") as f:
        [dev_data] = pickle.load(f)

dev_data = Dataset.from_dict(dev_data)
dev_data = dev_data.map(tokenize, batched=False)
dev_data.set_format(type="torch")

with open('empathy_score_dev_output.txt', 'w') as dev_f:
    accuracy_list = []
    for i in range(len((dev_data))):
        input_ids = dev_data["input_ids"][i]
        outputs = model.generate(**input_ids, do_sample=True, max_new_tokens=40, use_cache=True)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        dev_f.write(f"Batch {i}\n")
        for j in range(len(input_ids)):
            # Compute BLEU score
            prompt = dev_data[i]["query"][j].get("prompt")
            target = dev_data[i]["query"][j].get("target")
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], generated_texts[j])
            accuracy_list.append(BLEUscore)
            dev_f.write(f"{j} Prompt: {prompt}\n")
            dev_f.write(f"{j} Response: {generated_texts[j]}\n")
            dev_f.write(f"{j} Ref: {target}\n")
            dev_f.write('\n')


dev_f.close()
