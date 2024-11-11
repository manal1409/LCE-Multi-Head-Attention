from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average="weighted")
    metrics['recall'] = recall_score(y_true, y_pred, average="weighted")
    metrics['f1_score'] = f1_score(y_true, y_pred, average="weighted")
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    return metrics
