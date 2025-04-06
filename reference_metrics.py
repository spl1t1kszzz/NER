import json

import LEA

texts = ['79830', '418701', '542718', '713920', '716918', '731102', '732240', '737018', '737046', '747330', '747488',
         '760298']
models = ['o3-mini', '4o', 'o3-mini-high']
prompt = 'new_reference_CoT'


def create_prompts(prompt, texts):
    with open(f"./prompts/reference/{prompt}.txt", 'r', encoding='utf-8') as f:
        prompt_text = f.read()
        for t in texts:
            with open(f"./texts/{t}/text_{t}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                a = prompt_text.replace('<ТВОЙ ТЕКСТ ЗДЕСЬ>', data['text'])
                with open(f'{t}.txt', 'w', encoding='utf-8') as f1:
                    f1.write(a)


def get_metrics_for_model(model, prompt, texts):
    f1_scores = []
    for text in texts:
        pred_clusters_filename = f'results/prompt/{prompt}/{model}/{text}.json'
        true_clusters_filename = f"./texts/{text}/text_{text}.json"
        f1 = LEA.calculate_metrics(pred_clusters_filename, true_clusters_filename)['f1']
        f1_scores.append(f1)
    print('avg F1 score', model, prompt, sum(f1_scores) / len(f1_scores))


prompts = ['reference_CoT', 'new_reference_CoT', '2_new_ref_CoT']
for p in prompts:
    get_metrics_for_model('4o', p, texts)