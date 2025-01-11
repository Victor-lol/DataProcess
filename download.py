
from datasets import load_dataset


dataset_name = 'deepmind/aqua_rat'
save_path = './Data/AQRA-RAT'

dataset = load_dataset(dataset_name, ignore_verifications=True)
dataset.save_to_disk(save_path)