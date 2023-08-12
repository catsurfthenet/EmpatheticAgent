import pickle
from datasets import load_dataset
from datasets import Dataset, DatasetDict

from helper import get_emo_counts, get_js_distance, load_emo_classifier
import torch

top_count = 10
threshold = 0.4
load_path_prefix = '../'
set = "train"

size = 5000
train_size = 2000
val_size = 1000
test_size = 2000
train_dataset_path = f"{load_path_prefix}modeldata/sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_prosocial_dataset.p"#sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_EDOS_ED_dataset.p"
with open(train_dataset_path, "rb") as f:
    [data] = pickle.load(f)
ps_ds = Dataset.from_dict(data)
ps_ds = ps_ds.shuffle(seed=2023).select(range(size))

train_dataset_path = f"{load_path_prefix}modeldata/sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_chitchat_dataset.p"#sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_EDOS_ED_dataset.p"
with open(train_dataset_path, "rb") as f:
    [data] = pickle.load(f)
cc_ds = Dataset.from_dict(data)
cc_ds = cc_ds.shuffle(seed=2023).select(range(size))

train_dataset_path = f"{load_path_prefix}modeldata/sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_EDOS_ED_dataset.p"#sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_EDOS_ED_dataset.p"
with open(train_dataset_path, "rb") as f:
    [data] = pickle.load(f)
edos_ed_ds = Dataset.from_dict(data)
edos_ed_ds = edos_ed_ds.shuffle(seed=2023).select(range(size))


set = ["train", "validation", "test"]
start = [0, 2000, 3000]
end = [2000, 3000, 5000]
for i in range(len(set)):
    target = []
    new_prompt = []
    prompt_emo = []
    target_emo = []

    target += ps_ds["target"][start[i]:end[i]]
    target += cc_ds["target"][start[i]:end[i]]
    target += edos_ed_ds["target"][start[i]:end[i]]

    new_prompt += ps_ds["prompt"][start[i]:end[i]]
    new_prompt += cc_ds["prompt"][start[i]:end[i]]
    new_prompt += edos_ed_ds["prompt"][start[i]:end[i]]

    prompt_emo += ps_ds["prompt_emo"][start[i]:end[i]]
    prompt_emo += cc_ds["prompt_emo"][start[i]:end[i]]
    prompt_emo += edos_ed_ds["prompt_emo"][start[i]:end[i]]

    target_emo += ps_ds["target_emo"][start[i]:end[i]]
    target_emo += cc_ds["target_emo"][start[i]:end[i]]
    target_emo += edos_ed_ds["target_emo"][start[i]:end[i]]


    dict = {"prompt" : new_prompt,
            "target" : target,
            "prompt_emo": prompt_emo,
            "target_emo": target_emo}

    #dataset = Dataset.from_dict(dict)
    # local: '../', remote: ''
    f = open(f"{load_path_prefix}modeldata/MIX_sp_token_ws_empathy_clean_count_top{top_count}_score{threshold}_emo_{set[i]}_dataset.p", 'wb')# _dis_score
    pickle.dump([dict], f)
    f.close()