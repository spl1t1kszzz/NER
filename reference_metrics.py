import json
import os

import pandas as pd
from tqdm import tqdm
from typing import List
import LEA
from openai import OpenAI


def create_prompts(reference: bool, prompt, texts):
    prompt_template_file_name = f"./prompts/reference/{prompt}.txt" if reference else f"./prompts/term_classification/{prompt}.txt"
    with open(prompt_template_file_name, 'r', encoding='utf-8') as template:
        prompt_text = template.read()
        for t in tqdm(texts, desc=f"Создание промптов {prompt}"):
            with open(f"./texts/{t}/text_{t}.json", 'r', encoding='utf-8') as text:
                data = json.load(text)
                # data = text.read()
                a = prompt_text.replace('<ТВОЙ ТЕКСТ ЗДЕСЬ>', data['text'])
                os.makedirs(f'./ready_prompts/{prompt}', exist_ok=True)
                with open(f'./ready_prompts/{prompt}/{t}.txt', 'w', encoding='utf-8') as f1:
                    f1.write(a)


def get_metrics_for_model(model, prompt, texts):
    all_metrics = []

    for text in texts:
        pred_clusters_filename = f'results/prompt/{prompt}/{model}/{text}.json'
        true_clusters_filename = f"./texts/{text}/text_{text}.json"

        if not (os.path.exists(pred_clusters_filename) and os.path.exists(true_clusters_filename)):
            print(f"[!] Пропущен текст {text} — отсутствует один из файлов")
            continue
        metrics = LEA.calculate_metrics(pred_clusters_filename, true_clusters_filename)
        all_metrics.append(metrics)

    if not all_metrics:
        print("[!] Не найдено ни одного набора метрик.")
        return pd.DataFrame()

    keys = all_metrics[0].keys()
    avg_metrics = {f"avg_{key}": sum(m[key] for m in all_metrics if key in m) / len(all_metrics) for key in keys}

    df = pd.DataFrame([{
        "Модель": model,
        "Промпт": prompt,
        **{k.replace("avg_", "").capitalize(): round(v, 3) for k, v in avg_metrics.items()}
    }])

    return df


def resolve_reference(model_, prompt, texts):
    key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    print(models_map[model_])
    for t in tqdm(texts, desc="Обработка текстов"):
        with open(f"./ready_prompts/{prompt}/{t}.txt", 'r', encoding='utf-8') as prompt_file:
            prompt_text = prompt_file.read()
            response = client.chat.completions.create(
                model=models_map[model_],
                messages=[
                    {"role": "system",
                     "content": "Ты являешься экспертом в задаче решения референций из текстов на русском языке."},
                    {"role": "user", "content": prompt_text}
                ]
            )
            answer = response.choices[0].message.content.strip()
            result = {"clusters": answer}
            with open(f"./results/prompt/{prompt}/{model_}/{t}.json", 'w', encoding='utf-8') as json_file:
                # json_file.write(answer)
                json.dump(result, json_file, ensure_ascii=False, indent=4)


models_map = {
    '4o-mini-tuned': 'ft:gpt-4o-mini-2024-07-18:personal:reftuning:BJaiuVY9',
    '4o-mini-tuned-rucoco': 'ft:gpt-4o-mini-2024-07-18:personal:rucoco-fine-tuning:BPkqdkZt',
    '4,1-mini': 'gpt-4.1-mini-2025-04-14',
    '4,1-mini-tuned-rucoco': 'ft:gpt-4.1-mini-2025-04-14:personal:rucoco-coref:BQ8Xu1hn',
    '4,1': 'gpt-4.1-2025-04-14',
    '4,1-mini-tuned': 'ft:gpt-4.1-mini-2025-04-14:personal:ref-tuning:BTopNuPM'}
prompt_template = 'reference_one_shot'
texts = ['79830', '418701', '542718', '731102', '737018', '737046', '747330', '747488',
         '760298']
m = ['4o', '4,1', '4o-mini-tuned', '4,1-mini-tuned']
# create_prompts(True, prompt_template, texts)
model = m[0]
# resolve_reference(model, prompt_template, texts)
# df = get_metrics_for_model(model, prompt_template, texts)
# print(df)
# frames = []
# for p in ['reference_one_shot', 'reference_zero_shot', '2_new_ref_CoT', '3_new_ref_CoT', '3_new_ref_CoT_updated', '3_new_ref_CoT_updated_BIO']:
#     for model in m:
#         print(model)
#         # resolve_reference(model, prompt_template, texts)
#         df = get_metrics_for_model(model, p, texts)
#         if not df.empty:
#             frames.append(df)
#
# final_df = pd.concat(frames, ignore_index=True)
# final_df.to_csv(f"res.csv", index=False)
# print(final_df)


