
from datasets import load_dataset


dataset_name = 'shangzhu/ChemQA-lite'
save_path = './Data/ChemQA'

dataset = load_dataset(dataset_name, ignore_verifications=True)
dataset.save_to_disk(save_path)