from fcntl import FASYNC

from click import prompt


def get_prompt_template(prompt_file_name: str) -> str:
    with open(prompt_file_name, 'r', encoding='utf8') as prompt_file:
        return prompt_file.read()


def get_bio_from_response(response_file_name: str) -> list[str]:
    bio = []
    with open(response_file_name, 'r', encoding='utf8') as response_file:
        response = response_file.read()
        for token_label in response.split('\n'):
            bio.append(token_label.split()[1])
    return bio


def get_bio_from_dataset(dataset_file_name: str) -> list[str]:
    bio = []
    with open(dataset_file_name, 'r', encoding='utf8') as dataset_file:
        dataset = dataset_file.readlines()

    # Обрабатываем строки для извлечения BIO-разметки
    for line in dataset:
        # Пропускаем строки, которые начинаются с "# text" или пустые строки
        if line.startswith('# text') or not line.strip():
            continue

        # Разбиваем строку на слово и метку
        parts = line.strip().split()
        if len(parts) == 2:
            word, label = parts
            bio.append(label)
        else:
            print(line, len(parts))

    return bio


def run(num, gpt: bool, prompt_file_name: str, resp_file_name: str):
    if gpt:
        prompt_template = get_prompt_template(prompt_file_name)

        with open(f'./new_datasets/dataset_{num}/dataset_{num}.txt', 'r', encoding='utf8') as dataset_:
            dataset = dataset_.read()
            sentences = dataset.split('# text = ')
            for sentence in sentences:
                if sentence != '':
                    text = sentence.strip().split('\n')[0]
                    prompt = prompt_template.replace('{текст пользователя}', text)
                    # print(text)
                    from openai import OpenAI

                    client = OpenAI(
                        api_key="sk-or-vv-b346be4981c98e325b76eec4189e90682506e01b4ce1b7466cfb01774dacae52",
                        base_url="https://api.vsegpt.ru/v1",
                    )

                    messages = []
                    # messages.append({"role": "system", "content": system_text})
                    messages.append({"role": "user", "content": prompt})
                    # print(messages)

                    response_big = client.chat.completions.create(
                        model="openai/gpt-4",
                        messages=messages,
                        temperature=0.0,
                        n=1,
                        max_tokens=3000,
                    )

                    # print("Response BIG:",response_big)
                    response = response_big.choices[0].message.content
                    print(response)
    else:

        # resp = get_bio_from_response(f'./new_datasets/dataset_{num}/nodef+fp_corr.txt')
        resp = get_bio_from_response(resp_file_name)
        data = get_bio_from_dataset(f'./new_datasets/dataset_{num}/dataset_{num}.txt')

        print(len(resp), len(data))

        assert len(resp) == len(data)
        print(data)
        print(resp)

        from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

        y_true = [data]

        y_pred = [resp]

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Печать результатов
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Полный отчёт
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))


dataset_num = 1
gpt = False

run(dataset_num, gpt, prompt_file_name='./prompts/term_classification(def + CoT).txt',
    resp_file_name=f'./new_datasets/dataset_{dataset_num}/def + CoT.txt')
