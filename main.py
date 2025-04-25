# from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
# from openai import OpenAI
#
#
def get_prompt_template(prompt_file_name: str) -> str:
    with open(prompt_file_name, 'r', encoding='utf8') as prompt_file:
        return prompt_file.read()

#
# def get_bio_from_response(response_file_name: str) -> list[str]:
#     bio = []
#     with open(response_file_name, 'r', encoding='utf8') as response_file:
#         response = response_file.read()
#         for token_label in response.split('\n'):
#             bio.append(token_label.split()[1])
#
#     return bio
#
#
# def get_bio_from_dataset(dataset_file_name: str) -> list[str]:
#     bio = []
#     with open(dataset_file_name, 'r', encoding='utf8') as dataset_file:
#         dataset = dataset_file.readlines()
#
#     # Обрабатываем строки для извлечения BIO-разметки
#     for line in dataset:
#         # Пропускаем строки, которые начинаются с "# text" или пустые строки
#         if line.startswith('# text') or line.startswith('# relations') or not line.strip():
#             continue
#
#         # Разбиваем строку на слово и метку
#         parts = line.strip().split()
#         if len(parts) == 2:
#             word, label = parts
#             if '_' in label and not 'ML' in label:
#                 label = 'O'
#             bio.append(label)
#         else:
#             print(line, len(parts))
#
#     return bio
# #
# #
# # def run(num, gpt: bool, prompt_file_name: str, resp_file_name: str):
# #     if gpt:
# #         prompt_template = get_prompt_template(prompt_file_name)
# #
# #         with open(f'./new_datasets/dataset_{num}/dataset_entity_{num}.txt', 'r', encoding='utf8') as dataset_:
# #             dataset = dataset_.read()
# #             sentences = dataset.split('# text = ')
# #             for sentence in sentences:
# #                 if sentence != '':
# #                     text = sentence.strip().split('\n')[0]
# #                     prompt = prompt_template.replace('{текст пользователя}', text)
# #                     # print(text)
# #
# #                     client = OpenAI(
# #                         api_key="sk-or-vv-b346be4981c98e325b76eec4189e90682506e01b4ce1b7466cfb01774dacae52",
# #                         base_url="https://api.vsegpt.ru/v1",
# #                     )
# #
# #                     messages = []
# #                     # messages.append({"role": "system", "content": system_text})
# #                     messages.append({"role": "user", "content": prompt})
# #                     # print(messages)
# #
# #                     response_big = client.chat.completions.create(
# #                         model="mistralai/mixtral-8x7b-instruct",
# #                         messages=messages,
# #                         temperature=0.0,
# #                         n=1,
# #                         max_tokens=3000,
# #                     )
# #
# #                     # print("Response BIG:",response_big)
# #                     response = response_big.choices[0].message.content
# #                     # print(response)
# #                     with open(resp_file_name, 'a', encoding='utf8') as resp_file:
# #                         resp_file.write(response)
# #     else:
# #
# #         # resp = get_bio_from_response(f'./new_datasets/dataset_{num}/nodef+fp_corr.txt')
# #         resp = get_bio_from_response(resp_file_name)
# #         data = get_bio_from_dataset(f'./new_datasets/dataset_{num}/dataset_entity_{num}.txt')
# #
# #         print(len(resp), len(data))
# #
# #         assert len(resp) == len(data)
# #         print(data)
# #         print(resp)
# #
# #         y_true = [data]
# #
# #         y_pred = [resp]
# #
# #         precision = precision_score(y_true, y_pred)
# #         recall = recall_score(y_true, y_pred)
# #         f1 = f1_score(y_true, y_pred)
# #
# #         print(f"Precision: {precision:.2f}")
# #         print(f"Recall: {recall:.2f}")
# #         print(f"F1 Score: {f1:.2f}")
# #
# #         print("\nClassification Report:")
# #         print(classification_report(y_true, y_pred))
# #
# #
# # tricks = ['def', 'nodef', 'nodef + CoT', 'nodef + fp_corr', 'def + fp_corr', 'def + CoT']
# # # nums = [1, 2, 3, 4, 5, 7, 11, 15, 18, 21, 23, 26, 27, 39, 41, 43, 44, 45, 46, 47, 48, 49, 54, 62, 64, 66, 67, 69, 72,
# # #         74, 75, 76, 78, 84, 85, 91, 93, 94, 95, 96, 97, 98, 99]
# # nums = [1]
# # for num in nums:
# #     gpt = False
# #     print(num, 'start')
# #     for i in range(3, 4):
# #         if gpt:
# #             with open(f'./new_datasets/dataset_{num}/{tricks[i - 1]}.txt', 'w') as response_file:
# #                 response_file.close()
# #         run(num, gpt, prompt_file_name=f'./prompts/term_classification({tricks[i - 1]}).txt',
# #             resp_file_name=f'./new_datasets/dataset_{num}/{tricks[i - 1]}.txt')
# #     print(num, 'end')
# # #
# # #
# # #
# from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
#
# from collections import defaultdict
#
#
# from collections import defaultdict
#
# def extract_entities_by_class(bio_tags):
#     """
#     Извлекает сущности из BIO-разметки и группирует их по классам.
#     """
#     entities = defaultdict(list)
#     entity_start = None
#     entity_type = None
#
#     for i, tag in enumerate(bio_tags):
#         if tag == "O":
#             if entity_start is not None:
#                 entities[entity_type].append((entity_start, i - 1))
#                 entity_start = None
#                 entity_type = None
#         elif tag.startswith("B-"):
#             if entity_start is not None:
#                 entities[entity_type].append((entity_start, i - 1))
#             entity_start = i
#             entity_type = tag[2:]
#         elif tag.startswith("I-") and entity_type == tag[2:]:
#             continue
#         else:
#             if entity_start is not None:
#                 entities[entity_type].append((entity_start, i - 1))
#             entity_start = None
#             entity_type = None
#
#     if entity_start is not None:
#         entities[entity_type].append((entity_start, len(bio_tags) - 1))
#
#     return entities
#
#
# def compute_class_and_global_metrics(y_true_bio, y_pred_bio):
#     """
#     Вычисляет метрики (Precision, Recall, F1) для каждого класса и общие метрики.
#     """
#     # Извлекаем сущности по классам
#     true_entities = extract_entities_by_class(y_true_bio)
#     pred_entities = extract_entities_by_class(y_pred_bio)
#
#     # Метрики по классам
#     metrics = {}
#
#     all_true_set = set()
#     all_pred_set = set()
#
#     for entity_type in set(true_entities.keys()).union(pred_entities.keys()):
#         true_set = set(true_entities.get(entity_type, []))
#         pred_set = set(pred_entities.get(entity_type, []))
#
#         all_true_set.update(true_set)
#         all_pred_set.update(pred_set)
#
#         tp = len(true_set & pred_set)
#         fp = len(pred_set - true_set)
#         fn = len(true_set - pred_set)
#
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
#
#         metrics[entity_type] = {
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "true_entities": len(true_set),
#             "pred_entities": len(pred_set),
#             "correct_entities": tp,
#         }
#
#     # Общие метрики
#     tp_global = len(all_true_set & all_pred_set)
#     fp_global = len(all_pred_set - all_true_set)
#     fn_global = len(all_true_set - all_pred_set)
#
#     precision_global = tp_global / (tp_global + fp_global) if (tp_global + fp_global) > 0 else 0.0
#     recall_global = tp_global / (tp_global + fn_global) if (tp_global + fn_global) > 0 else 0.0
#     f1_global = (2 * precision_global * recall_global) / (precision_global + recall_global) if (precision_global + recall_global) > 0 else 0.0
#
#     global_metrics = {
#         "precision": precision_global,
#         "recall": recall_global,
#         "f1": f1_global,
#         "true_entities": len(all_true_set),
#         "pred_entities": len(all_pred_set),
#         "correct_entities": tp_global,
#     }
#
#     return metrics, global_metrics
#
#
# # Инициализация для всех датасетов
# all_true_bio = []
# all_pred_bio = []
#
# # Список номеров датасетов
# # dataset_nums = [1, 2, 3, 4, 5, 7, 11, 15, 18, 21, 23, 26, 27, 39, 41, 43, 44, 45, 46, 47, 48, 49, 54, 62, 64, 66, 67, 69, 72, 74, 75, 76, 78, 84, 85, 91, 93, 94, 95, 96, 97, 98, 99]  # Замените на ваши номера датасетов
# dataset_nums = [1]
# for num in dataset_nums:
#     # Получение разметок BIO
#     resp_file_name = f'./new_datasets/dataset_{num}/nodef + CoT.txt'  # Путь к предсказаниям
#     resp = get_bio_from_response(resp_file_name)
#     data = get_bio_from_dataset(f'./new_datasets/dataset_{num}/dataset_entity_{num}.txt')
#
#     # Проверка длины
#     assert len(resp) == len(data), f"Несоответствие длины предсказаний и данных для датасета {num}"
#
#     # Добавление данных
#     all_true_bio.extend(data)
#     all_pred_bio.extend(resp)
#
# # Вычисление метрик
# class_metrics, global_metrics = compute_class_and_global_metrics(all_true_bio, all_pred_bio)
#
# # Вывод метрик по классам
# # print(f"Метрики по классам сущностей:")
# # for entity_type, metrics in class_metrics.items():
# #     print(f"Класс: {entity_type}")
# #     print(f"  Precision: {metrics['precision']:.2f}")
# #     print(f"  Recall: {metrics['recall']:.2f}")
# #     print(f"  F1 Score: {metrics['f1']:.2f}")
# #     # print(f"  Истинных сущностей: {metrics['true_entities']}")
# #     # print(f"  Предсказанных сущностей: {metrics['pred_entities']}")
# #     # print(f"  Верно предсказанных сущностей: {metrics['correct_entities']}\n")
#
# # Вывод общих метрик
# print(f"Общие метрики по всем сущностям:")
# print(f"Precision: {global_metrics['precision']:.2f}")
# print(f"Recall: {global_metrics['recall']:.2f}")
# print(f"F1 Score: {global_metrics['f1']:.2f}")
# print(f"Истинных сущностей: {global_metrics['true_entities']}")
# print(f"Предсказанных сущностей: {global_metrics['pred_entities']}")
# print(f"Верно предсказанных сущностей: {global_metrics['correct_entities']}")


import json

from openai import OpenAI
# tricks = ['def', 'nodef', 'nodef + CoT', 'nodef + fp_corr', 'def + fp_corr', 'def + CoT']
# prompt_file_name=f'./prompts/term_classification({tricks[len(tricks) - 1]}).txt'
# with open('docIE_cl_ext.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#     for article in data['articles']:
#         for text in article:
#             for t in article[text]:
#                 print(t)
#                 # prompt_template = get_prompt_template(prompt_file_name)
#                 # prompt = prompt_template.replace('{текст пользователя}', t['text'])
#                 # client = OpenAI(
#                 #     api_key="sk-or-vv-b346be4981c98e325b76eec4189e90682506e01b4ce1b7466cfb01774dacae52",
#                 #     base_url="https://api.vsegpt.ru/v1",
#                 # )
#                 # messages = []
#                 # messages.append({"role": "user", "content": prompt})
#                 # response_big = client.chat.completions.create(
#                 #     model="mistralai/mixtral-8x7b-instruct",
#                 #     messages=messages,
#                 #     temperature=0.0,
#                 #     n=1,
#                 #     max_tokens=3000,
#                 # )
#                 # response = response_big.choices[0].message.content
#                 # print(response)
#                 # with open(f'{tricks[len(tricks) - 1]}.txt', 'a', encoding='utf8') as resp_file:
#                 #     resp_file.write(response)





