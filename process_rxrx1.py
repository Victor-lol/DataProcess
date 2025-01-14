
import os 
import requests
import torch
from PIL import Image
import pandas as pd 

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def construct_data(df, root):
  messages = []
  pbar = tqdm(len(df), total=len(df))

  for idx, row in df.iterrows():
      id_code = row['well_id']
      site = row['site']
      path = id_code.partition('_')[0] + '/Plate' + id_code.partition('_')[2][0] + '/'+ id_code.rpartition('_')[2]
      img_path1 = os.path.join(root, f'{path}_s{site}_w1.png')
      img_path2 = os.path.join(root, f'{path}_s{site}_w2.png')
      img_path3 = os.path.join(root, f'{path}_s{site}_w3.png')
      img_path4 = os.path.join(root, f'{path}_s{site}_w4.png')
      img_path5 = os.path.join(root, f'{path}_s{site}_w5.png')
      img_path6 = os.path.join(root, f'{path}_s{site}_w6.png')
      message = {
          'role': 'user',
          'content': [
              {'type': 'image', 'image': img_path1},
              {'type': 'image', 'image': img_path2},
              {'type': 'image', 'image': img_path3},
              {'type': 'image', 'image': img_path4},
              {'type': 'image', 'image': img_path5},
              {'type': 'image', 'image': img_path6},
              {'type': 'text', 'text': fixed_prompt},
            ]
          }
      messages.append(message)
      pbar.update(1)
  pbar.close()

  return messages

def load_model(model_id="Qwen/Qwen2-VL-7B-Instruct"):

    if 'qwen' in model_id.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
           model_id, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    return model, processor 

def single_inference_qwen(model, processor, prompt, gen_config):

    prompt = [prompt]
    temperature = gen_config['temperature']
    max_new_tokens = gen_config['max_new_tokens']

    text = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(prompt)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

if __name__ == '__main__':

    metadata_path = '/content/drive/MyDrive/NIPS_benchmark/RxRx1/metadata.csv'
    model_id = 'Qwen/Qwen2-VL-7B-Instruct'
    metadata = pd.read_csv(metadata_path)
    root = '/content/drive/MyDrive/NIPS_benchmark/RxRx1/rxrx1/rxrx1/images'
    test = metadata[metadata['dataset'] == 'test']

    prompts = construct_data(test)
    model, processor = load_model(model_id=model_id)
    gen_config = {
        'temperature': .6,
        'max_new_tokens': 32
    }

    output = single_inference_qwen(model, processor, prompts[0], gen_config)