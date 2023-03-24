import os 
import requests

os.mkdir('squad')

url = "https://rajpurkar.github.io/SQuAD-explorer/dataset"

for file in ['train-v2.0.json', 'dev-v2.0.json']:
    res = requests.get(f'{url}{file}')
