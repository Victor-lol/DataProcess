
import json 
from datasets import load_from_disk
import random 
import csv
from tqdm import tqdm
import os

# path = 'Data/AQRA-RAT'
path = 'Data/PubMedQA'
dataset = load_from_disk(path)

# Save the list of dictionaries to a CSV file
def save_to_csv(data, file_name):
    # Extract the keys for the CSV header from the first dictionary
    keys = data[0].keys()
    
    with open(file_name, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        
        # Write the header
        writer.writeheader()
        
        # Write the data rows
        writer.writerows(data)

def process_pubmed(dataset):
    train = dataset['train']
    test = dataset['test']
    train_list, test_list = [], []
    root = './ProcessedData/PubMedQA'

    os.makedirs(root, exist_ok=True)
    
    for row in tqdm(train):
        question = row['input']
        label = row['output']
        inputs = row['instruction']

        train_list.append({
            'ContextString': inputs, 
            'Behavior': question, 
            'answer': label
        })
    for row in tqdm(test):
        question = row['input']
        label = row['output']
        inputs = row['instruction']

        test_list.append({
            'ContextString': inputs, 
            'Behavior': question, 
            'answer': label
        })
    
    train_list = random.sample(train_list, k=400)
    test_list = random.sample(test_list, k=400)
    
    save_to_csv(train_list, os.path.join(root, 'train_aqua.csv'))
    save_to_csv(test_list, os.path.join(root, 'train_aqua.csv'))

def process_AQUA(dataset):

    train = dataset['train']
    test = dataset['test']
    train_list, test_list = [], []
    root = './ProcessedData/AQUA'

    os.makedirs(root, exist_ok=True)
    
    for row in tqdm(train):
        question = row['question']
        label = row['correct']
        inputs = row['options']

        train_list.append({
            'ContextString': inputs, 
            'Behavior': question, 
            'answer': label
        })
    for row in tqdm(test):
        question = row['question']
        label = row['correct']
        inputs = row['options']

        test_list.append({
            'ContextString': inputs, 
            'Behavior': question, 
            'answer': label
        })
    
    train_list = random.sample(train_list, k=400)
    test_list = random.sample(test_list, k=400)


    save_to_csv(train_list, os.path.join(root, 'train_aqua.csv'))
    save_to_csv(test_list, os.path.join(root, 'train_aqua.csv'))
    


# process_AQUA(dataset)
process_pubmed(dataset)
