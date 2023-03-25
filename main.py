# Import required libraries
import os
import requests
import json
import torch
from transformers import DistilBertTokenizerFast

# Create a new directory for SQuAD dataset
"""
os.mkdir('squad')

# Set URL and download SQuAD dataset files (train and dev)
url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
for file in ['train-v2.0.json', 'dev-v2.0.json']:
    res = requests.get(f'{url}{file}')
    with open(f'squad/{file}','wb') as f:
        for chunk in res.iter_content(chunk_size=4):
            f.write(chunk)
"""
# Function to read SQuAD dataset from a given path
def read_squad(path):
    with open('squad/train-v2.0.json', 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []

# Extract context, questions and answers from the dataset
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

# Read training and validation sets
train_contexts, train_questions, train_answers = read_squad('squad/train-v2.json')
val_contexts, val_questions, val_answers = read_squad('squad/dev-v2.json')

# Function to add end indices to answers
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

# Add end indices for both training and validation sets
add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize training and validation sets
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

# Function to add token positions for start and end of the answers
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end']-1)
    encodings.update({
        'start_positions': start_positions,
        'end_positions': end_positions
    })

# Add token positions to train and validation encodings
add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

# Print the first 10 start positions of training encodings
print(train_encodings['start_positions'][:10])