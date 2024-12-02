from anyio import sleep
from click import prompt
from openai import OpenAI

client = OpenAI(
    api_key="sk-or-vv-b346be4981c98e325b76eec4189e90682506e01b4ce1b7466cfb01774dacae52",
    # ваш ключ в VseGPT после регистрации
    base_url="https://api.vsegpt.ru/v1",
)


def get_text(file_name: str):
    with open(file_name, "r", encoding='utf-8') as f:
        content = f.read()
    sections = content.split('\n\n')
    data = []
    for i, section in enumerate(sections):
        line = sections[i].split('\n')[0]
        text = line.replace("# text = ", "").strip()
        data.append(text)
    return data


ent_class = "Environment"
texts = get_text("./new_datasets/dataset_2/dataset_2.txt")
for text in texts:
    prompt = fr'Извлеки именованные сущности, принадлежащие только классу {ent_class} из следующего текста: "{text}"'
    print(prompt)
    # with open("./prompts/system_prompt.txt", 'r', encoding='utf-8') as f:
    #     system_text = f.read()
    # messages = [{"role": "system", "content": system_text}, {"role": "user", "content": prompt}]
    # response_big = client.chat.completions.create(
    #     model="openai/gpt-3.5-turbo",
    #     messages=messages,
    #     temperature=0.0,
    #     n=1,
    #     max_tokens=3000,
    #     extra_headers={"X-Title": "My App"},
    # )
    #
    # response = response_big.choices[0].message.content
    # print("Response:", response)
    # print("**********")
