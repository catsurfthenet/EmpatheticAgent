import csv

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, \
    pipeline

pretrained_model = "facebook/blenderbot-400M-distill"
emp_classifier_model = "../models/roberta-empathy-03-06-2023-18_21_58"

empathy_model_id = emp_classifier_model
empathy_classifier = pipeline('text-classification', model=empathy_model_id, tokenizer=empathy_model_id, max_length=512, truncation=True, top_k=None)

# read texts
with open("../modeldata/manual_empathetic.csv", 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    prompt = []
    utterance = []
    score = []
    for row in csvreader:
        if not row[2].isnumeric(): # check if its column name using score
            continue

        prompt.append(row[0])
        utterance.append(row[1])
        score.append(row[2])
file.close()

