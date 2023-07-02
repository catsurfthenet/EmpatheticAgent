import matplotlib
import os
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset

emo_model_id = "SamLowe/roberta-base-go_emotions"
#tokenizer = AutoTokenizer.from_pretrained(emo_model_id)
#model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
emo_classifier = pipeline('text-classification', model=emo_model_id,tokenizer=emo_model_id, max_length=512, truncation=True, top_k=None)

if(os.path.exists('/modeldata/empathetic_dialogues_original_dataset.p')):
    print("LOADING data")
    with open('/modeldata/empathetic_dialogues_original_dataset.p', "rb") as f:
        [dataset] = pickle.load(f)
    f.close()
else:
    dataset = load_dataset("empathetic_dialogues")


train_results = emo_classifier(dataset["train"]["utterance"])

dev_results = emo_classifier(dataset["validation"]["utterance"])

test_results = emo_classifier(dataset["test"]["utterance"])

#f = open("modeldata/emo_classified_dataset.p", 'wb')
#pickle.dump([train_results, dev_results, test_results], f)
#f.close()

