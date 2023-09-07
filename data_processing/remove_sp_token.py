import pickle
from datasets import Dataset, DatasetDict
import re

# code for removing special tokens from datasets created with special tokens
load_path_prefix = "../"
set = "validation"
train_dataset_path = f"{load_path_prefix}modeldata/sp_token_context_ws_empathy_clean_count_top10_score0.4_emo_{set}_ED_dataset.p"#sp_token_ws_empathy_clean_count_top10_score0.4_emo_train_EDOS_ED_dataset.p"
with open(train_dataset_path, "rb") as f:
    [data] = pickle.load(f)
ed_ds = Dataset.from_dict(data)

# initialise
prompt = ed_ds["prompt"]
target = ed_ds["target"]
prompt_emo = ed_ds["prompt_emo"]
target_emo = ed_ds["target_emo"]
dia_context = ed_ds["context"]
clean_prompt = []
clean_target = []
for i in ed_ds:
    p = i["prompt"]
    pe = i["prompt_emo"]
    # replace the [emotion] part with empty string ""
    p = p.replace(f"[{pe}] ", "")
    t = i["target"]
    te = i["target_emo"]
    # replace the [emotion] part with empty string ""
    t = t.replace(f"[{te}] ", "")
    clean_prompt.append(p)
    clean_target.append(t)

# create a dict for saving
dict = {"prompt": clean_prompt,
        "target": clean_target,
        "prompt_emo": prompt_emo,
        "target_emo": target_emo,
        "context": dia_context}

# save the resulting dataset
f = open(f"../modeldata/context_ws_empathy_clean_count_top10_score0.4_emo_{set}_ED_dataset.p", 'wb')
pickle.dump([dict], f)
f.close()
