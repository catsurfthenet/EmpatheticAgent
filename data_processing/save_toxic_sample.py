import pickle

from datasets import load_dataset, Dataset

# save 1000 data sample from dataset "allenai/real-toxicity-prompts" on huggingface
neg_sample_size = 1000
toxic_ds = load_dataset("allenai/real-toxicity-prompts", split="train").shuffle(seed=2023).select(range(neg_sample_size))
toxic_ds = Dataset.from_list(toxic_ds["prompt"])["text"]

f = open(f"../modeldata/real_toxicity_prompts_1000.p", 'wb')
pickle.dump([toxic_ds], f)
f.close()