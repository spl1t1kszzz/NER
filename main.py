# import json
# import os
#
#
def read_json_file(file_name: str):
    with open(file_name, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        return json_data

# import json
# from sklearn.metrics import precision_score, recall_score, f1_score
#
# def extract_entities(data):
#     true_entities = []
#     pred_entities = []
#
#     for item in data:
#         # Для истинных меток
#         for entity in item['entities']:
#             true_entities.append((item['text'][entity['start']:entity['end']], entity['label']))
#
#         # Для предсказанных меток (здесь просто пример, замените на ваши предсказания)
#         # В реальном использовании у вас должны быть предсказанные аннотации
#         # Пример: [{"start": 10, "end": 16, "label": "PERSON"}]
#         # Для демонстрации предполагаем, что предсказанные значения совпадают с истинными.
#         pred_entities.append((item['text'][entity['start']:entity['end']], entity['label']))
#
#     return true_entities, pred_entities
#
# def calculate_metrics(true_entities, pred_entities):
#     # Извлекаем только метки
#     true_labels = [label for _, label in true_entities]
#     pred_labels = [label for _, label in pred_entities]
#
#     # Рассчитаем метрики
#     precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
#     recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
#     f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
#
#     return precision, recall, f1
#
# # Пример JSON данных
# json_data = [
#     {
#         "text": "Я зовут Алексей.",
#         "entities": [{"start": 10, "end": 16, "label": "PERSON"}]
#     },
#     {
#         "text": "Она работает в Microsoft.",
#         "entities": [{"start": 23, "end": 30, "label": "ORG"}]
#     }
# ]
#
# # Извлекаем сущности
# true_entities, pred_entities = extract_entities(json_data)
#
# # Рассчитываем метрики
# precision, recall, f1 = calculate_metrics(true_entities, pred_entities)
#
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")

def read_file(file_name):
    with open(file_name, 'r') as f:
        return f.read()


from openai import OpenAI
import json

dataset_num = 3
text_index = 2
text = read_json_file(f'./dataset_{dataset_num}/entities.json')[text_index]['text']

client = OpenAI(
    api_key="sk-or-vv-b346be4981c98e325b76eec4189e90682506e01b4ce1b7466cfb01774dacae52",
    base_url="https://api.vsegpt.ru/v1",
)

prompt = f'''Извлеки именованные сущности из следующего текста и классифицируй их по следующим категориям:
Object
Metric
Task
Model
Dataset
Organization
Science
Person
Publication
Method
Application (Library, App_system, Technology)
Environment
InfoResource (Corpus)
Activity
Текст:{text}'''
print(prompt)

messages = [{"role": "system", "content": read_file("system_prompt.txt")}, {"role": "user", "content": prompt}]

response_big = client.chat.completions.create(
    model="OMF-R-IlyaGusev/saiga_llama3_8b",
    messages=messages,
    temperature=0.0,
    n=1,
    max_tokens=3000,
    extra_headers={"X-Title": "My App"},
)

# print("Response BIG:",response_big)
response = response_big.choices[0].message.content
print("Response:", response)
