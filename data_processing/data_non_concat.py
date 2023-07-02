import pickle
from datasets import load_dataset
from datasets import Dataset, DatasetDict

dataset = load_dataset("empathetic_dialogues")

set = "test"

prompt = dataset[set]["prompt"]
utterance = dataset[set]["utterance"]
conv_id = ""
prev_context = ""
target = []
new_prompt = []

for d in dataset[set]:
    if d["conv_id"] != conv_id:
        conv_id = d["conv_id"]
        prev_context = d["utterance"]
    else:
        new_prompt.append(prev_context)
        target.append(d["utterance"])
        prev_context = d["utterance"]

dict = {"prompt" : new_prompt,
        "target" : target}

#dataset = Dataset.from_dict(dict)
f = open("../modeldata/test_dialogue_dataset.p", 'wb')
pickle.dump([dict], f)
f.close()

"""
# checks
print(new_prompt[:7])
print(target[:7])
print(len(new_prompt) == len(target))
"""
