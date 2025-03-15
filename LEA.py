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
