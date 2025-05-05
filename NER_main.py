from tqdm import tqdm

import NER_metrics
from NER_metrics import *
import os


def solve_ner_old_dataset(num, prompt_file_name: str, resp_file_name: str, model_: str):
    prompt_template = get_prompt_template(prompt_file_name)

    with open(f'./old_datasets/dataset_{num}/dataset_entity_{num}.txt', 'r', encoding='utf8') as dataset_:
        dataset = dataset_.read()
        sentences = dataset.split('# text = ')
        for sentence in sentences:
            if sentence != '':
                text = sentence.strip().split('\n')[0]
                prompt = prompt_template.replace('{текст пользователя}', text)
                key = os.getenv("OPENAI_API_KEY")
                client = OpenAI(api_key=key)
                response = client.chat.completions.create(
                    model=models_map[model_],
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content.strip()

                with open(resp_file_name, 'a', encoding='utf8') as resp_file:
                    resp_file.write(answer)


def solve_ner_new_dataset(dataset_name: str, prompt_file_name: str, resp_file_name: str, model_: str):
    prompt_template = get_prompt_template(prompt_file_name)

    with open(f'./new_datasets/{dataset_name}.json', 'r', encoding='utf8') as dataset_:
        examples = json.loads(dataset_.read())['examples']
        for sentence in tqdm(examples, desc=dataset_name):
            text = sentence['text']
            prompt = prompt_template.replace('{текст пользователя}', text)
            key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model=models_map[model_],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()

            with open(resp_file_name, 'a', encoding='utf8') as resp_file:
                resp_file.write(answer)


model = '4o'
prompt = 'nodef + fp_corr'
metrics_sum = {'Recall': 0.0, 'Precision' : 0.0, 'F1' : 0.0}
datasets = ['paragraph_dataset', 'new_dataset', 'new_dataset_2']
for dataset in datasets:
    # solve_ner_new_dataset(dataset, f'./prompts/term_classification/term_classification({prompt}).txt', f'./new_datasets/results/{prompt}/{model}/{dataset}.txt', model)
    m = NER_metrics.get_ner_metrics(False, 0, f'./new_datasets/{dataset}_BIO.txt',f'./new_datasets/results/{prompt}/{model}/{dataset}.txt')
    metrics_sum['Recall'] += m['recall']
    metrics_sum['Precision'] += m['precision']
    metrics_sum['F1'] += m['f1']
metrics_sum['Recall'] = round(metrics_sum['Recall'] / len(datasets), 3)
metrics_sum['Precision'] = round(metrics_sum['Precision'] / len(datasets), 3)
metrics_sum['F1'] = round(metrics_sum['F1'] / len(datasets), 3)

print('AVG metrics', metrics_sum)
