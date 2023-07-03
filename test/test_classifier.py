import csv
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, \
    pipeline, Trainer, RobertaTokenizerFast, RobertaForSequenceClassification
from datasets import Dataset
from sklearn.metrics import mean_squared_error as mse
from scipy.special import logit
import numpy as np
import evaluate

pretrain_choice = 1
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

# Custom torch dataset
class custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(eval_preds):
    metric = evaluate.load("mse", "f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# read texts
with open("../modeldata/manual_empathetic.csv", 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    prompt = []
    utterance = []
    score_list = []
    for row in csvreader:
        if not row[2].isnumeric(): # check if its column name using score
            continue

        prompt.append(row[0])
        utterance.append(row[1])
        score_list.append(row[2])
file.close()

pretrained_model = "facebook/blenderbot-400M-distill"
emp_classifier_model = "../models/roberta-empathy-03-06-2023-18_21_58"
true = [int(s)/7 for s in score_list]

if pretrain_choice == 1:
    empathy_model_id = emp_classifier_model
    empathy_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 512)
    empathy_model = RobertaForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32).to(device)
    empathy_classifier = pipeline('text-classification', model=empathy_model, tokenizer=empathy_tokenizer)
    results = empathy_classifier(utterance, padding='max_length', truncation=True, max_length=512)#, labels=true
    #test_trainer = Trainer(
    #    model=empathy_classifier,
    #    compute_metrics=compute_metrics)
    # Make prediction
    #raw_preds, _, _ = test_trainer.predict(results)
    # Preprocess raw predictions
    #y_preds = np.argmax(raw_preds, axis=1)

else:
    empathy_model_id = "bdotloh/roberta-base-empathy"
    empathy_tokenizer = AutoTokenizer.from_pretrained(empathy_model_id)
    empathy_model = AutoModelForSequenceClassification.from_pretrained(empathy_model_id, torch_dtype=torch.float32)
    empathy_classifier = pipeline('text-classification', model=empathy_model_id)
    results = Dataset.from_list(empathy_classifier(utterance))


#print(results)
pred = []
accurate_num = 0
for i in range(len(results)):
    #true.append(actual)
    actual = true[i]
    if actual == 0:
        actual += 1e-15
        true[i] = actual
    elif actual == 1:
        actual -= 1e-15
        true[i] = actual

    if pretrain_choice == 0:
        if results[i]["label"] == "Empathy":
            pred.append(results[i]['score'])
            print(f"{results[i]}, {actual}")
        else:
            score = results[i]['score']
            pred.append((1-score))
            print(f"{results[i]}, {1-score}, {actual}")
    elif pretrain_choice == 1:
        # LABEL_0: no empathy, LABEL_1: weak empathy, LABEL_2: strong empathy
        labels = []
        for s in score_list:
            if int(s) <= 4:
                labels.append("LABEL_0")
            elif int(s) == 5:
                labels.append("LABEL_1")
            else:
                labels.append("LABEL_2")
        if results[i]['label'] == labels[i]:
            accurate_num += 1
        print(f"{results[i]}, {labels[i]}")

accuracy = accurate_num / len(results)
print(f"Accuracy: {accuracy}")
#MSE_loss = mse(true, pred)
#MSE_loss_logit = mse(logit(true), logit(pred))
#print(f"MSE loss: {MSE_loss}, MSE logit loss: {MSE_loss_logit}")
#test_dataset = Dataset(results)

