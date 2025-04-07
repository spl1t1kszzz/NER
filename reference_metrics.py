import json
import os
import tiktoken
from tqdm import tqdm
from typing import List
import LEA
from openai import OpenAI


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
    with open(f"./prompts/reference/{prompt}.txt", 'r', encoding='utf-8') as template:
        prompt_text = template.read()
        for t in tqdm(texts, desc=f"Создание промптов {prompt}"):
            with open(f"./texts/{t}/text_{t}.json", 'r', encoding='utf-8') as text:
                data = json.load(text)
                a = prompt_text.replace('<ТВОЙ ТЕКСТ ЗДЕСЬ>', data['text'])
                os.makedirs(f'./ready_prompts/{prompt}', exist_ok=True)
                with open(f'./ready_prompts/{prompt}/{t}.txt', 'w', encoding='utf-8') as f1:
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
    all_metrics = []

    for text in texts:
        pred_clusters_filename = f'results/prompt/{prompt}/{model}/{text}.json'
        true_clusters_filename = f"./texts/{text}/text_{text}.json"
        metrics = LEA.calculate_metrics(pred_clusters_filename, true_clusters_filename)
        all_metrics.append(metrics)
    keys = all_metrics[0].keys()
    avg_metrics = {}

    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        avg_metrics[f"avg_{key}"] = sum(values) / len(values) if values else 0.0

    return avg_metrics


models = {'4o': 'gpt-4o-2024-08-06', '4o-mini': 'gpt-4o-mini-2024-07-18',
          '4o-mini-tuned': 'ft:gpt-4o-mini-2024-07-18:personal:reftuning:BJaiuVY9'}


def resolve_reference(model_, prompt, texts):
    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    print(models[model_])
    for t in tqdm(texts, desc="Обработка текстов"):
        with open(f"./ready_prompts/{prompt}/{t}.txt", 'r', encoding='utf-8') as prompt_file:
            prompt_text = prompt_file.read()
            response = client.chat.completions.create(
                model=models[model_],
                messages=[
                    {"role": "system",
                     "content": "Ты являешься экспертом в задаче решения референций из текстов на русском языке."},
                    {"role": "user", "content": prompt_text}
                ]
            )
            answer = response.choices[0].message.content.strip()
            result = {"clusters": answer}
            with open(f"./results/prompt/{prompt}/{model_}/{t}.json", 'w', encoding='utf-8') as json_file:
                json.dump(result, json_file, ensure_ascii=False, indent=4)


prompt_template = '3_new_ref_CoT'
texts = ['79830', '418701', '542718', '731102', '737018', '737046', '747330', '747488',
         '760298']
m = ['4o', '4o-mini', '4o-mini-tuned']
# create_prompts(prompt_template, texts)
# resolve_reference(model, prompt_template, texts)
for model in m:
    print(model, get_metrics_for_model(model, prompt_template, texts)['avg_f1'])
