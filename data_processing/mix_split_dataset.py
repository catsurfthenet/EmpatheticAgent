import pickle
from datasets import load_dataset
from datasets import Dataset, DatasetDict

from helper import get_emo_counts, get_js_distance, load_emo_classifier
import torch

# code for creating a mixed dataset using multiple datasets
top_count = 10
threshold = 0.4
load_path_prefix = '../'
set = "train"
seed = 42

size = 1000 # total size = 6000
r = [3,1,1,1] #ratio
train_size = 5000#3000
val_size = 1000#1000
test_size = 10000#2000


train_dataset_path = f"{load_path_prefix}modeldata/sp_token_ws_empathy_clean_count_top10_score0.4_emo_AnnoMI_dataset.p"#sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_EDOS_ED_dataset.p"
with open(train_dataset_path, "rb") as f:
    [data] = pickle.load(f)
edos_ed_ds = Dataset.from_dict(data)
#edos_ed_ds = edos_ed_ds.shuffle(seed=seed).select(range(train_size)) #r[0] * size

"""
train_dataset_path = f"{load_path_prefix}modeldata/sp_token_ws_empathy_clean_count_top10_score0.4_emo_medical_dataset.p"#sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_EDOS_ED_dataset.p"
with open(train_dataset_path, "rb") as f:
    [data] = pickle.load(f)
md_ds = Dataset.from_dict(data)
md_ds = md_ds.shuffle(seed=seed).select(range(r[1] *size))

train_dataset_path = f"{load_path_prefix}modeldata/sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_prosocial_dataset.p"#sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_EDOS_ED_dataset.p"
with open(train_dataset_path, "rb") as f:
    [data] = pickle.load(f)
ps_ds = Dataset.from_dict(data)
ps_ds = ps_ds.shuffle(seed=seed).select(range(r[2] *size))

train_dataset_path = f"{load_path_prefix}modeldata/sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_chitchat_dataset.p"#sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_EDOS_ED_dataset.p"
with open(train_dataset_path, "rb") as f:
    [data] = pickle.load(f)
cc_ds = Dataset.from_dict(data)
cc_ds = cc_ds.shuffle(seed=seed).select(range(r[3] *size))
"""

set = ["train", "validation"] # "test"
ed_start = [0, 2000, 2234]
ed_end = [2000, 2235, 2235]

#ed_start = [0, 1500, 2000]
#ed_end = [1500, 2000, 3000]
start = [0, 500, 667]
end = [500, 667, 1000]
for i in range(len(set)):
    target = []
    new_prompt = []
    prompt_emo = []
    target_emo = []

    #target += ps_ds["target"][start[i]:end[i]]
    #target += md_ds["target"][start[i]:end[i]]
    #target += cc_ds["target"][start[i]:end[i]]
    target += edos_ed_ds["target"][ed_start[i]:ed_end[i]]

    #new_prompt += ps_ds["prompt"][start[i]:end[i]]
    #new_prompt += md_ds["prompt"][start[i]:end[i]]
    #new_prompt += cc_ds["prompt"][start[i]:end[i]]
    new_prompt += edos_ed_ds["prompt"][ed_start[i]:ed_end[i]]

    #prompt_emo += ps_ds["prompt_emo"][start[i]:end[i]]
    #prompt_emo += md_ds["prompt_emo"][start[i]:end[i]]
    #prompt_emo += cc_ds["prompt_emo"][start[i]:end[i]]
    prompt_emo += edos_ed_ds["prompt_emo"][ed_start[i]:ed_end[i]]

    #target_emo += ps_ds["target_emo"][start[i]:end[i]]
    #target_emo += md_ds["target_emo"][start[i]:end[i]]
    #target_emo += cc_ds["target_emo"][start[i]:end[i]]
    target_emo += edos_ed_ds["target_emo"][ed_start[i]:ed_end[i]]


    dict = {"prompt" : new_prompt,
            "target" : target,
            "prompt_emo": prompt_emo,
            "target_emo": target_emo}

    #dataset = Dataset.from_dict(dict)
    # local: '../', remote: ''
    f = open(f"{load_path_prefix}modeldata/sp_token_ws_empathy_clean_count_top{top_count}_score{threshold}_emo_{set[i]}_AM_dataset.p", 'wb')# _dis_score
    pickle.dump([dict], f)
    f.close()