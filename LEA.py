import json


def lea(clusters_true, clusters_pred):
    def compute_links(cluster):
        n = len(cluster)
        return n * (n - 1) / 2

    def compute_intersection_links(cluster_pred_, cluster_true):
        intersection = set(cluster_pred_).intersection(set(cluster_true))
        return compute_links(intersection)

    precision_numerator = 0
    precision_denominator = 0

    for cluster_pred in clusters_pred:
        if len(cluster_pred) <= 1:
            continue
        precision_denominator += len(cluster_pred)
        precision_numerator += len(cluster_pred) * sum(
            compute_intersection_links(cluster_pred, cluster_true) / compute_links(cluster_pred)
            for cluster_true in clusters_true)

    recall_numerator = 0
    recall_denominator = 0

    for cluster_true in clusters_true:
        if len(cluster_true) <= 1:
            continue
        recall_denominator += len(cluster_true)
        recall_numerator += len(cluster_true) * sum(
            compute_intersection_links(cluster_true, cluster_pred) / compute_links(cluster_true)
            for cluster_pred in clusters_pred)

    precision = precision_numerator / precision_denominator if precision_denominator > 0 else 0
    recall = recall_numerator / recall_denominator if recall_denominator > 0 else 0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def calculate_metrics(pred_clusters_filename, true_clusters_filename):
    with open(pred_clusters_filename, "r", encoding="utf-8") as file:
        predicted = json.load(file)['clusters']
    clusters_true = get_true_clusters(true_clusters_filename)
    return lea(clusters_true, predicted)


def get_true_clusters(clusters_filename):
    with open(clusters_filename, "r", encoding="utf-8") as true_clusters_file:
        data = json.load(true_clusters_file)
        text = data['text']
        clusters_true = data['entities']
        clusters_with_mentions = []
        for cluster in clusters_true:
            mentions = [text[start:end] for start, end in cluster]
            clusters_with_mentions.append(mentions)
        return clusters_with_mentions
