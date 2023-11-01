import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def run_classificatoin_report(model, loader):

    # Switch the model to evaluation mode
    model.eval()

    # Initialize lists to store true labels and model predictions
    true_labels = []
    model_predictions = []

    # No need for gradient computation here
    with torch.no_grad():
        for inputs, labels in loader:
            # Forward pass
            outputs = model(inputs)
            
            # Assuming the output of the model is a raw score for each class
            # and we take the score of the positive class or the class with the highest score
            _, predicted_classes = torch.max(outputs, 1)
            
            # Store true labels and predictions for evaluation
            true_labels.extend(labels.numpy())
            model_predictions.extend(predicted_classes.numpy())

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, model_predictions)
    display_labels = [str(lbl) for lbl in sorted(set(true_labels))]  # Create a list of string labels

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels).plot(cmap='bwr', values_format='d', ax=ax)
    plt.title('Confusion Matrix')
    plt.show()

    # Generate a classification report
    report = classification_report(true_labels, model_predictions, target_names=display_labels, output_dict=True)

    # Convert the classification report to a DataFrame for easy plotting
    import pandas as pd
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