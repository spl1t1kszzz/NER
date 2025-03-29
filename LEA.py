import json


def lea(clusters_true, clusters_pred):
    def compute_links(cluster):
        n = len(cluster)
        return n * (n - 1) / 2

    def compute_intersection_links(cluster_pred_, cluster_true):
        intersection = len(set(cluster_pred_).intersection(set(cluster_true)))
        return intersection * (intersection - 1) / 2

    precision_numerator = 0
    precision_denominator = 0

    for cluster_pred in clusters_pred:
        if len(cluster_pred) <= 1:
            continue
        precision_denominator += compute_links(cluster_pred)
        precision_numerator += sum(compute_intersection_links(cluster_pred, cluster_true)
                                   for cluster_true in clusters_true)

    recall_numerator = 0
    recall_denominator = 0

    for cluster_true in clusters_true:
        if len(cluster_true) <= 1:
            continue
        recall_denominator += compute_links(cluster_true)
        recall_numerator += sum(compute_intersection_links(cluster_true, cluster_pred)
                                for cluster_pred in clusters_pred)

    precision = precision_numerator / precision_denominator if precision_denominator > 0 else 0
    recall = recall_numerator / recall_denominator if recall_denominator > 0 else 0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


texts = ['79830', '418701', '542718', '713920', '716918', '731102', '732240', '737018', '737046', '747330', '747488',
         '760298']
f1_sum = 0
for text_id in texts:
    with open(f"./texts/{text_id}/text_{text_id}.json", "r", encoding="utf-8") as text_file:
        data = json.load(text_file)

    text = data['text']
    clusters_true = data['entities']

    clusters_with_mentions = []

    for cluster in clusters_true:
        mentions = []
        for start, end in cluster:
            mention = text[start:end]
            mentions.append(mention)
        clusters_with_mentions.append(mentions)
    with open(f"./texts/{text_id}/ref_CoT.json", "r", encoding="utf-8") as entities_file:
        data = json.load(entities_file)
        res = data['entities']
        f1_sum += data['metrics']['f1']
print(f1_sum / len(texts))
