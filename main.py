import os 
import requests
import json
from transformers import DistilBertTokenizerFast

"""
os.mkdir('squad')

url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

for file in ['train-v2.0.json', 'dev-v2.0.json']:
    res = requests.get(f'{url}{file}')
    with open(f'squad/{file}','wb') as f:
        for chunk in res.iter_content(chunk_size=4):
            f.write(chunk)
"""
# Data Prep

def read_squad(path):
    with open('squad/train-v2.0.json', 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []

    for group in squad_dict['data']:
        for passage in group["paragraphs"]:  
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'  
                for answer in qa[access]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


train_contexts, train_questions, train_answers = read_squad('squad/train-v2.json')
val_contexts, val_questions, val_answers = read_squad('squad/dev-v2.json')

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        else:
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n

add_end_idx(train_answers, train_contexts)   
add_end_idx(val_answers, val_contexts)   

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

print(train_encodings.keys())