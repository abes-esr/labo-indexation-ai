"""Utilitary functions used for text processing in ABES project"""


# Import librairies
import numpy as np

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    jaccard_score,
)
from sklearn.metrics import (
    hamming_loss,
    brier_score_loss,
    classification_report,
)


from utils_text_processing import *


# Compute metrics
def label_metrics_report(
    modelName,
    y_true,
    y_pred,
    y_prob=None,
    classes=None,
    zero_division="warn",
    print_metrics=False,
):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(
        y_true, y_pred, average="macro", zero_division=zero_division
    )
    macro_recall = recall_score(
        y_true, y_pred, average="macro", zero_division=zero_division
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=zero_division)
    macro_jaccard = jaccard_score(
        y_true, y_pred, average="macro", zero_division=zero_division
    )
    micro_precision = precision_score(
        y_true, y_pred, average="micro", zero_division=zero_division
    )
    micro_recall = recall_score(
        y_true, y_pred, average="micro", zero_division=zero_division
    )
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=zero_division)
    micro_jaccard = jaccard_score(
        y_true, y_pred, average="micro", zero_division=zero_division
    )
    sample_precision = precision_score(
        y_true, y_pred, average="samples", zero_division=zero_division
    )
    sample_recall = recall_score(
        y_true, y_pred, average="samples", zero_division=zero_division
    )
    sample_f1 = f1_score(y_true, y_pred, average="samples", zero_division=zero_division)
    sample_jaccard = jaccard_score(
        y_true, y_pred, average="samples", zero_division=zero_division
    )
    hamLoss = hamming_loss(y_true, y_pred)
    if y_prob:
        brier = brier_score_loss(y_true, y_prob)
    else:
        brier = np.nan

    if print_metrics:
        # Print result
        print("------" + modelName + " Model Metrics-----")
        print(
            f"Accuracy: {accuracy:.4f}\nHamming Loss: {hamLoss:.4f}\Brier score Loss: {brier:.4f}"
        )
        print(
            f"Precision:\n  - Macro: {macro_precision:.4f}\n  - Micro: {micro_precision:.4f}"
        )
        print(f"Recall:\n  - Macro: {macro_recall:.4f}\n  - Micro: {micro_recall:.4f}")
        print(f"F1-measure:\n  - Macro: {macro_f1:.4f}\n  - Micro: {micro_f1:.4f}")
        print(
            f"Jaccard similarity:\n  - Macro: {macro_jaccard:.4f}\n  - Micro: {micro_jaccard:.4f}"
        )
        classification_report(y_true, y_pred, target_names=classes)

    return {
        "Hamming Loss": hamLoss,
        "Brier Loss": brier,
        "Accuracy": accuracy,
        "Precision - Macro": macro_precision,
        "Recall - Macro": macro_recall,
        "F1_Score - Macro": macro_f1,
        "Jaccard - Macro": macro_jaccard,
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
