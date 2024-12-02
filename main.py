from collections import defaultdict

import nltk
from sklearn.metrics import precision_score, recall_score, f1_score
from ner_eval import Evaluator, collect_named_entities, compute_metrics, compute_precision_recall_wrapper


def extract_entities(tokens, labels):
    entities = []
    current_entity = []
    current_entity_type = None

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            if current_entity:  # если мы уже собирали сущность
                entities.append((current_entity_type, " ".join(current_entity)))
                current_entity = []
            current_entity_type = label[2:]  # Тип сущности
            current_entity.append(token)
        elif label.startswith("I-") and current_entity:
            current_entity.append(token)  # продолжаем собирать сущность
        else:
            if current_entity:  # если мы закончили собирать сущность
                entities.append((current_entity_type, " ".join(current_entity)))
                current_entity = []
                current_entity_type = None

    if current_entity:  # добавляем последнюю сущность, если есть
        entities.append((current_entity_type, " ".join(current_entity)))

    return entities


def get_labels(file_name: str):
    with open(file_name, 'r', encoding='utf-8') as file:
        content = file.read()
    sections = content.split('\n\n')
    labels = []
    for i, section in enumerate(sections):
        line = sections[i].split('\n')[0]
        text = line.replace("# text = ", "").strip()
        tokens = nltk.word_tokenize(text, language='russian')
        data = section.split('\n')[1:]
        section_labels = [element.split()[-1] for element in data]
        labels.extend(section_labels)
    return labels


def calculate_metrics_for_class(dataset_num: int, ent_class: str):
    true_labels = get_labels(f"new_datasets/dataset_{dataset_num}/dataset_{dataset_num}.txt")
    pred_labels = get_labels(f"new_datasets/dataset_{dataset_num}/all_classes_without_def.txt")

    true_named_entities_type = defaultdict(list)
    pred_named_entities_type = defaultdict(list)

    for true in collect_named_entities(true_labels):
        true_named_entities_type[true.e_type].append(true)

    for pred in collect_named_entities(pred_labels):
        pred_named_entities_type[pred.e_type].append(pred)

    results, results_agg = compute_metrics(true_named_entities_type[ent_class], pred_named_entities_type[ent_class],
                              tags=[ent_class])
    results = compute_precision_recall_wrapper(results)
    return results


def calculate_metrics_for_all_classes(dataset_num: int):
    true_labels = get_labels(f"new_datasets/dataset_{dataset_num}/dataset_{dataset_num}.txt")
    pred_labels = get_labels(f"new_datasets/dataset_{dataset_num}/all_classes_with_def.txt")

    results, results_agg = compute_metrics(collect_named_entities(true_labels), collect_named_entities(pred_labels),
                                           tags=set([label[2:] for label in true_labels if len(label) > 2]))
    results = compute_precision_recall_wrapper(results)
    return results

res = calculate_metrics_for_all_classes(4)
for r in res:
    print(r, res[r])
