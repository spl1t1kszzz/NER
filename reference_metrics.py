import LEA

texts = ['79830', '418701', '542718', '713920', '716918', '731102', '732240', '737018', '737046', '747330', '747488',
         '760298']
models = ['o3-mini', '4o']
prompt = 'reference_CoT'
for model in models:
    f1_scores = []
    for text in texts:
        pred_clusters_filename = f'results/prompt/{prompt}/{model}/{text}.json'
        true_clusters_filename = f"./texts/{text}/text_{text}.json"
        f1 = LEA.calculate_metrics(pred_clusters_filename, true_clusters_filename)['f1']
        f1_scores.append(f1)
    print('avg F1 score', model, prompt, sum(f1_scores) / len(f1_scores))
