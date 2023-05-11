"""Utilitary functions used for text processing in ABES project"""


# Import librairies
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    jaccard_score,
)
from sklearn.metrics import (
    hamming_loss,
    confusion_matrix,
    multilabel_confusion_matrix,
    classification_report,
)

from sklearn.metrics import (
    coverage_error,
    label_ranking_average_precision_score,
    label_ranking_loss,
)
from sklearn.metrics import precision_recall_fscore_support as score

from utils_text_processing import *


# Compute cosine similarity for 2 sentences
def cosine_similarity(doc1, doc2):
    y1 = nlp(doc1)
    y2 = nlp(doc2)
    cos_sim = y1.similarity(y2)

    return cos_sim


# Compute cosine similarity for 2 indexeurs
def cosine_similarity(index1, index2):
    cos = []
    for doc1, doc2 in zip(index1, index2):
        print(f"comparing {doc1} and {doc2}")
        cos_sim = cosine_similarity(doc1, doc2)
        cos.append(cos_sim)

    return cos.mean()


# Compute metrics
def label_metrics_report(modelName, y_true, y_pred, print_metrics=False):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_jaccard = jaccard_score(y_true, y_pred, average="macro")
    micro_precision = precision_score(y_true, y_pred, average="micro")
    micro_recall = recall_score(y_true, y_pred, average="micro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    micro_jaccard = jaccard_score(y_true, y_pred, average="micro")
    sample_precision = precision_score(y_true, y_pred, average="samples")
    sample_recall = recall_score(y_true, y_pred, average="samples")
    sample_f1 = f1_score(y_true, y_pred, average="samples")
    sample_jaccard = jaccard_score(y_true, y_pred, average="samples")
    hamLoss = hamming_loss(y_true, y_pred)

    if print_metrics:
        # Print result
        print("------" + modelName + " Model Metrics-----")
        print(
            f"Accuracy: {accuracy:.4f}\nHamming Loss: {hamLoss:.4f}\nCosine Similarity: {cos_simi:.4f}"
        )
        print(
            f"Precision:\n  - Macro: {macro_precision:.4f}\n  - Micro: {micro_precision:.4f}"
        )
        print(f"Recall:\n  - Macro: {macro_recall:.4f}\n  - Micro: {micro_recall:.4f}")
        print(f"F1-measure:\n  - Macro: {macro_f1:.4f}\n  - Micro: {micro_f1:.4f}")
        print(
            f"Jaccard similarity:\n  - Macro: {macro_jaccard:.4f}\n  - Micro: {micro_jaccard:.4f}"
        )
    return {
        "Hamming Loss": hamLoss,
        "Accuracy": accuracy,
        "Precision - Micro": micro_precision,
        "Recall - Micro": micro_recall,
        "F1_Score - Micro": micro_f1,
        "Jaccard - Micro": micro_jaccard,
        "Precision - Sample": sample_precision,
        "Recall - Sample": sample_recall,
        "F1_Score - Sample": sample_f1,
        "Jaccard - Sample": sample_jaccard,
        "Precision": {
            "Macro": macro_precision,
            "Micro": micro_precision,
            "Sample": sample_precision,
        },
        "Recall": {
            "Macro": macro_recall,
            "Micro": micro_recall,
            "Sample": sample_recall,
        },
        "F1-measure": {"Macro": macro_f1, "Micro": micro_f1, "Sample": sample_f1},
        "Jaccard": {
            "Macro": macro_jaccard,
            "Micro": micro_jaccard,
            "Sample": sample_jaccard,
        },
    }
