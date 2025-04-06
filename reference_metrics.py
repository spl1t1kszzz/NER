import json
import os
from typing import List

import tiktoken

import LEA

texts = ['79830', '418701', '542718', '713920', '716918', '731102', '732240', '737018', '737046', '747330', '747488',
         '760298']
models = ['o3-mini', '4o', 'o3-mini-high']
prompt = 'new_reference_CoT'


def split_text_by_tokens(text: str, model: str = "gpt-4o", max_tokens: int = 1000, overlap: int = 200) -> List[str]:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        i += max_tokens - overlap

    return chunks


def create_prompts(prompt, texts):
    with open(f"./prompts/reference/{prompt}.txt", 'r', encoding='utf-8') as f:
        prompt_text = f.read()
        for t in texts:
            with open(f"./texts/{t}/text_{t}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                a = prompt_text.replace('<ТВОЙ ТЕКСТ ЗДЕСЬ>', data['text'])
                with open(f'{t}.txt', 'w', encoding='utf-8') as f1:
                    f1.write(a)


def create_context_prompts(prompt, texts, model):
    with open(f"./prompts/reference/{prompt}.txt", 'r', encoding='utf-8') as f:
        prompt_text = f.read()
        for t in texts:
            os.makedirs(f"./{t}", exist_ok=True)
            with open(f"./texts/{t}/text_{t}.json", 'r', encoding='utf-8') as text_file:
                data = json.load(text_file)['text']
                split_data = split_text_by_tokens(data, model, max_tokens=1000)
                for idx, chunk in enumerate(split_data):
                    with open(f'./{t}/{idx}.txt', 'w', encoding='utf-8') as f1:
                        if idx == 0:
                            p = prompt_text.replace('<ТВОЙ ТЕКСТ ЗДЕСЬ>', data)
                            f1.write(p)
                        else:
                            f1.write(chunk)


def get_metrics_for_model(model, prompt, texts):
    f1_scores = []
    for text in texts:
        pred_clusters_filename = f'results/prompt/{prompt}/{model}/{text}.json'
        true_clusters_filename = f"./texts/{text}/text_{text}.json"
        f1 = LEA.calculate_metrics(pred_clusters_filename, true_clusters_filename)['f1']
        f1_scores.append(f1)
    print('avg F1 score', model, prompt, sum(f1_scores) / len(f1_scores))


# prompts = ['reference_CoT', 'new_reference_CoT', '2_new_ref_CoT']
# for p in prompts:
#     get_metrics_for_model('4o', p, texts)


# get_metrics_for_model('4o', 'reference_CoT', texts)
# get_metrics_for_model('4o', 'new_reference_CoT', texts)
# get_metrics_for_model('4o', '2_new_ref_CoT', texts)
# get_metrics_for_model('4o', '3_new_ref_CoT', texts)
get_metrics_for_model('4o', 'ref_ctx', ['79830'])
print(LEA.get_true_clusters(f'./texts/79830/text_79830.json'))

create_context_prompts('ref_ctx', texts, 'gpt-4o')
