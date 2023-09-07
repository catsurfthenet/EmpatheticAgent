import os
import pickle
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from datasets import Dataset, DatasetDict

# load Empathetic Dialogues (ED) dataset
if(os.path.exists('modeldata/model_dataset.p')):
    print("LOADING data")
    with open('modeldata/model_dataset.p', "rb") as f:
        [gen_model, tokenizerOne, tokenizerTwo, cls_model, classifier, dataset] = pickle.load(f)
else:
    dataset = load_dataset("empathetic_dialogues")

# define tokenize function and tokenize datasets
def tokenize_function(examples):
    return tokenizerOne(examples["prompt"])

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch")

# add padding to the entries of the datasets
def padding(data):
    padded = pad_sequence(data)
    padded = list(map(torch.Tensor, padded.T))
    return padded

ids_train = padding(tokenized_datasets["train"]['input_ids'])
ids_dev = padding(tokenized_datasets["validation"]['input_ids'])
ids_test = padding(tokenized_datasets["test"]['input_ids'])


ah_train = padding(tokenized_datasets["train"]['attention_mask'])
ah_dev = padding(tokenized_datasets["validation"]['attention_mask'])
ah_test = padding(tokenized_datasets["test"]['attention_mask'])

# recreate the dataset using padded data
train_dict = Dataset.from_dict({"input_ids": ids_train, "attention_mask": ah_train})
dev_dict = Dataset.from_dict({"input_ids": ids_dev, "attention_mask": ah_dev})
test_dict = Dataset.from_dict({"input_ids": ids_test, "attention_mask": ah_test})

# create a dict for saving
dict = {"train": train_dict,
        "validation": dev_dict,
        "test": test_dict}
pro_dataset = DatasetDict(dict)

# save the resulting dataset
f = open("modeldata/processed_dataset.p", 'wb')
pickle.dump([pro_dataset], f)
f.close()