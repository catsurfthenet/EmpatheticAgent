import pickle
from datasets import Dataset

# training set
ed_dataset_path = '../modeldata/dialogue_dataset.p'
edos_dataset_path = '../modeldata/edos_non_concat_data.p'

with open(ed_dataset_path, "rb") as f:
    [ed_ds] = pickle.load(f)
ed_ds = Dataset.from_dict(ed_ds)
f.close()

with open(edos_dataset_path, "rb") as f2:
    [edos_prompt, edos_target] = pickle.load(f2)
f2.close()

ed_prompt = list(ed_ds["prompt"])
ed_target = list(ed_ds["target"])

new_prompt = list(edos_prompt) + ed_prompt
target = list(edos_target) + ed_target

dict = {"prompt" : new_prompt,
        "target" : target}

f3 = open("../modeldata/edos_dialogue_dataset.p", 'wb')
pickle.dump([dict], f3)
f3.close()



