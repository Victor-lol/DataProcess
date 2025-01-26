
from datasets import load_dataset


dataset_name = 'llamafactory/PubMedQA'
save_path = './Data/PubMedQA'

dataset = load_dataset(dataset_name, ignore_verifications=True)
print(len(dataset))
dataset.save_to_disk(save_path)