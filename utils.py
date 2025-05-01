import json
from typing import List
import pandas as pd
import os



def get_true_clusters(clusters_filename):
    with open(clusters_filename, "r", encoding="utf-8") as true_clusters_file:
        data = json.load(true_clusters_file)
        text = data['text']
        clusters_true = data['entities']
        clusters_with_mentions = []
        for cluster in clusters_true:
            mentions = [text[start:end] for start, end in cluster]
            clusters_with_mentions.append(mentions)
        return clusters_with_mentions


def create_fine_tuning_data(fine_tuning_file_name: str, prompt_template_file_name: str,
                            train_texts_file_names: List[str]):
    system_prompt = "Ты являешься экспертом в задаче решения референций из текстов на русском языке."

    with open(prompt_template_file_name, "r", encoding="utf-8") as prompt_template_file:
        prompt_template = prompt_template_file.read()

    with open(fine_tuning_file_name, "w", encoding="utf-8") as fine_tuning_file:
        for train_texts_file_name in train_texts_file_names:
            with open(train_texts_file_name, "r", encoding="utf-8") as train_texts_file:
                text = json.load(train_texts_file)['text']

            data_template = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_template.replace('<ТВОЙ ТЕКСТ ЗДЕСЬ>', text)},
                    {"role": "assistant", "content": str(get_true_clusters(train_texts_file_name))}
                ]
            }

            json_line = json.dumps(data_template, ensure_ascii=False)
            fine_tuning_file.write(json_line + "\n")


train_file_names = ['./train/2000_finance_mos.json', './train/2000_sport_asiancup_001.json',
                    './train/2000_world_peacetalk.json', './train/2001_cinema_mikhalkov.json',
                    './train/2001_russia_education_001.json', './train/2021_world_uscyberresponce.json',
                    './train/2020_hitech_ai_tax.json', './train/2020_hitech_anymalc_lake.json',
                    './train/2020_hitech_dit_viber.json', './train/2020_hitech_fake_vote.json']

# create_fine_tuning_data('./fine_tuning/10_texts_rucoco_fine_tuning.jsonl', './prompts/reference/3_new_ref_CoT.txt',
#                         train_file_names)



