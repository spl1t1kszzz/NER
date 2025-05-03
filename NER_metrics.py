import os
from collections import defaultdict
from models import models_map
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from openai import OpenAI
import re
import json
from typing import List, Dict, Tuple


def get_prompt_template(prompt_file_name: str) -> str:
    with open(prompt_file_name, 'r', encoding='utf8') as prompt_file:
        return prompt_file.read()


def get_bio_from_response(response_file_name: str) -> list[str]:
    bio = []
    with open(response_file_name, 'r', encoding='utf8') as response_file:
        response = response_file.read()
        for token_label in response.split('\n'):
            try:
                bio.append(token_label.split()[1])
            except IndexError:
                print(token_label)

    return bio


def get_bio_from_old_dataset(dataset_file_name: str) -> list[str]:
    bio = []
    with open(dataset_file_name, 'r', encoding='utf8') as dataset_file:
        dataset = dataset_file.readlines()

    for line in dataset:
        if line.startswith('# text') or line.startswith('# relations') or not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) == 2:
            word, label = parts
            if '_' in label and not 'ML' in label:
                label = 'O'
            bio.append(label)
        else:
            print(line, len(parts))

    return bio






def get_ner_metrics(is_old_dataset: bool, num: int, new_dataset_file_name:str, resp_file_name: str):
    resp = get_bio_from_response(resp_file_name)
    if is_old_dataset:
        data = get_bio_from_old_dataset(f'./old_datasets/dataset_{num}/dataset_entity_{num}.txt')
    else:
        data = get_bio_from_old_dataset(new_dataset_file_name)

    print(len(resp), len(data))

    assert len(resp) == len(data)
    print(data)
    print(resp)

    y_true = [data]

    y_pred = [resp]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {'precision': precision, 'recall': recall, 'f1': f1}


    # print("\nClassification Report:")
    # print(classification_report(y_true, y_pred))


def extract_entities_by_class(bio_tags):
    """
    Извлекает сущности из BIO-разметки и группирует их по классам.
    """
    entities = defaultdict(list)
    entity_start = None
    entity_type = None

    for i, tag in enumerate(bio_tags):
        if tag == "O":
            if entity_start is not None:
                entities[entity_type].append((entity_start, i - 1))
                entity_start = None
                entity_type = None
        elif tag.startswith("B-"):
            if entity_start is not None:
                entities[entity_type].append((entity_start, i - 1))
            entity_start = i
            entity_type = tag[2:]
        elif tag.startswith("I-") and entity_type == tag[2:]:
            continue
        else:
            if entity_start is not None:
                entities[entity_type].append((entity_start, i - 1))
            entity_start = None
            entity_type = None

    if entity_start is not None:
        entities[entity_type].append((entity_start, len(bio_tags) - 1))

    return entities


def compute_class_and_global_metrics(y_true_bio, y_pred_bio):
    """
    Вычисляет метрики (Precision, Recall, F1) для каждого класса и общие метрики.
    """
    # Извлекаем сущности по классам
    true_entities = extract_entities_by_class(y_true_bio)
    pred_entities = extract_entities_by_class(y_pred_bio)

    # Метрики по классам
    metrics = {}

    all_true_set = set()
    all_pred_set = set()

    for entity_type in set(true_entities.keys()).union(pred_entities.keys()):
        true_set = set(true_entities.get(entity_type, []))
        pred_set = set(pred_entities.get(entity_type, []))

        all_true_set.update(true_set)
        all_pred_set.update(pred_set)

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_entities": len(true_set),
            "pred_entities": len(pred_set),
            "correct_entities": tp,
        }

    # Общие метрики
    tp_global = len(all_true_set & all_pred_set)
    fp_global = len(all_pred_set - all_true_set)
    fn_global = len(all_true_set - all_pred_set)

    precision_global = tp_global / (tp_global + fp_global) if (tp_global + fp_global) > 0 else 0.0
    recall_global = tp_global / (tp_global + fn_global) if (tp_global + fn_global) > 0 else 0.0
    f1_global = (2 * precision_global * recall_global) / (precision_global + recall_global) if (
                                                                                                       precision_global + recall_global) > 0 else 0.0

    global_metrics = {
        "precision": precision_global,
        "recall": recall_global,
        "f1": f1_global,
        "true_entities": len(all_true_set),
        "pred_entities": len(all_pred_set),
        "correct_entities": tp_global,
    }

    return metrics, global_metrics


def simple_tokenize(text: str) -> List[Tuple[str, int, int]]:
    tokens = []
    for match in re.finditer(r'\w+|[^\w\s]', text, re.UNICODE):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def get_entity_offsets(text: str, terms: List[Dict[str, str]]) -> List[Tuple[int, int, str]]:
    offsets = []
    for term in terms:
        value = term['value']
        label = term['class']
        for match in re.finditer(re.escape(value), text):
            start, end = match.span()
            offsets.append((start, end, label))
    return sorted(offsets, key=lambda x: x[0])


def convert_to_bio(text: str, terms: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    tokens = simple_tokenize(text)
    bio_tags = ['O'] * len(tokens)

    entity_offsets = get_entity_offsets(text, terms)

    for start, end, label in entity_offsets:
        first_token_found = False
        for i, (tok, tok_start, tok_end) in enumerate(tokens):
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            if tok_start >= start and tok_end <= end:
                if not first_token_found:
                    bio_tags[i] = f'B-{label}'
                    first_token_found = True
                else:
                    bio_tags[i] = f'I-{label}'

    return [(tok, tag) for (tok, _, _), tag in zip(tokens, bio_tags)]


def convert_new_dataset_to_bio(json_data: Dict) -> List[Tuple[str, str]]:
    bio_lines = []
    for example in json_data["examples"]:
        bio_seq = convert_to_bio(example["text"], example["terms"])
        bio_lines.extend(bio_seq)
        bio_lines.append(("", ""))  # пустая строка между примерами
    return bio_lines


def save_bio_to_txt(bio_lines: List[Tuple[str, str]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for token, tag in bio_lines:
            if token == "" and tag == "":
                f.write("\n")
            else:
                f.write(f"{token}\t{tag}\n")

dataset_name = 'new_dataset_2'
# if __name__ == "__main__":
#     with open(f"./new_datasets/{dataset_name}.json", "r", encoding="utf-8") as f:
#         dataset = json.load(f)
#
#     bio_result = convert_new_dataset_to_bio(dataset)
#     save_bio_to_txt(bio_result, f"./new_datasets/{dataset_name}_BIO.txt")

