import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import pandas as pd


def compute_roc_auc_score(model, loader):
    # Switch the model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    # Initialize lists to store true labels and model predictions
    true_labels = []
    model_predictions = []

    model_scores = []  # To store the softmax scores for each class

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            softmax_scores = F.softmax(outputs, dim=1)  # Apply softmax to convert logits to probabilities
            _, predicted_classes = torch.max(softmax_scores, 1)
            
            # Store the softmax scores for ROC AUC computation
            model_scores.extend(softmax_scores.cpu().numpy())

            true_labels.extend(labels.cpu().numpy())
            model_predictions.extend(predicted_classes.cpu().numpy())

    # Ensure that there are more than two classes
    if len(set(true_labels)) > 2:
        true_labels_binarized = label_binarize(true_labels, classes=sorted(set(true_labels)))
        roc_auc = roc_auc_score(true_labels_binarized, model_scores, multi_class='ovr', average='macro')
    else:
        # If there are only two classes, do not binarize
        positive_class_scores = [score[1] for score in model_scores] # Use the scores for the positive class
        roc_auc = roc_auc_score(true_labels, positive_class_scores)
    return roc_auc

def run_classificatoin_report(model, loader, do_print=True):
    """
    Run a full classfication report.
    Return values:
    - Confusion matrix
    - Classification report
    - AUC-ROC score
    """

    # Switch the model to evaluation mode
    model.eval()

    # Initialize lists to store true labels and model predictions
    true_labels = []
    model_predictions = []

    model_scores = []  # To store the softmax scores for each class

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            softmax_scores = F.softmax(outputs, dim=1)  # Apply softmax to convert logits to probabilities
            _, predicted_classes = torch.max(softmax_scores, 1)
            
            # Store the softmax scores for ROC AUC computation
            model_scores.extend(softmax_scores.cpu().numpy())

            true_labels.extend(labels.numpy())
            model_predictions.extend(predicted_classes.numpy())

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, model_predictions)
    display_labels = [str(lbl) for lbl in sorted(set(true_labels))]  # Create a list of string labels

    # Plot the confusion matrix
    if do_print:
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels).plot(cmap='bwr', values_format='d', ax=ax)
        plt.title('Confusion Matrix')
        plt.show()

    # Generate a classification report
    report = classification_report(true_labels, model_predictions, target_names=display_labels, output_dict=True)

    if do_print:
        # Convert the classification report to a DataFrame for easy plotting
        report_df = pd.DataFrame(report).transpose()

        # Precision and recall for each class
        report_df = report_df.round(2)
        fig, ax = plt.subplots(figsize=(12, 6))

        # We skip the last rows which contain average values ('micro avg', 'macro avg', 'weighted avg')
        classes_report_df = report_df.loc[display_labels]
        classes_report_df.plot(kind='bar', y=['precision', 'recall'], ax=ax)
        plt.title('Precision and Recall for Each Class')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.xticks(rotation=45)  # Rotate class names for better legibility
        plt.show()

    # Ensure that there are more than two classes
    if len(set(true_labels)) > 2:
        true_labels_binarized = label_binarize(true_labels, classes=sorted(set(true_labels)))
        roc_auc = roc_auc_score(true_labels_binarized, model_scores, multi_class='ovr', average='macro')
    else:
        # If there are only two classes, do not binarize
        positive_class_scores = [score[1] for score in model_scores] # Use the scores for the positive class
        roc_auc = roc_auc_score(true_labels, positive_class_scores)  
    if do_print:
        print(f'Macro AUC-ROC score: {roc_auc}')

    return conf_matrix, report, roc_auc