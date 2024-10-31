import json
import os


def read_json_file(file_name: str):
    with open(file_name, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        return json_data


def calculate_metrics(true_ent, predicted_ent):
    metrics = {}
    for category in true_ent.keys():
        true_labels = [item['text'] for item in true_ent[category]]
        predicted_labels = [item['text'] for item in predicted_ent[category]]

        true_set = set(true_labels)
        predicted_set = set(predicted_labels)

        tp = len(true_set.intersection(predicted_set))  # Верные предсказания
        fp = len(predicted_set - true_set)  # Ложные срабатывания
        fn = len(true_set - predicted_set)  # Пропущенные метки

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        metrics[category] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    return metrics


def print_metrics(metrics):
    for category, metric_values in metrics.items():
        print(f"Категория: {category}")
        for metric_name, value in metric_values.items():
            print(f"  {metric_name}: {value:.2f}")
        print()


def get_true_entities(dataset_num: int):
    true_ent = read_json_file(f'./dataset_{dataset_num}/entities.json')
    return true_ent


def get_predicted_entities(dataset_num: int, res_num: int):
    pred_ent = read_json_file(f'./dataset_{dataset_num}/res_{res_num}.json')
    return pred_ent


def print_metrics_for_dataset(dataset_num: int):

    cur_dir = f'./dataset_{dataset_num}/'
    res_files = [f for f in os.listdir(cur_dir) if os.path.isfile(os.path.join(cur_dir, f)) and f.startswith('res')]
    tricks = read_json_file(f'./dataset_{dataset_num}/tricks.json')
    for res_file in res_files:
        print('---------------------------------------------------')
        res_num = os.path.splitext(res_file)[0][-1]
        print(tricks[res_num])
        true_entities = get_true_entities(dataset_num)
        predicted_entities = get_predicted_entities(dataset_num, res_num)
        met = calculate_metrics(true_entities, predicted_entities)
        print_metrics(met)
        print('---------------------------------------------------')


print_metrics_for_dataset(4)
