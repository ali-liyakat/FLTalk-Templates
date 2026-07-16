"""
Generates confusion matrix images from the evaluation results of different methods (Imbalanced, Local DeepSMOTE, FedDeepSMOTE).
The evaluation results are expected to be in a JSON file located at 'results_eval/mnist_eval_results.json'.
The generated confusion matrix images will be saved in the 'results_eval/cm_images/' directory.
"""



import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Correct path
with open("results_eval/mnist_eval_results.json", "r") as f:
    data = json.load(f)

# Output folder
os.makedirs("results_eval/cm_images", exist_ok=True)

def plot_cm(cm, title, filename):
    cm = np.array(cm)

    plt.figure()
    sns.heatmap(cm, annot=False, fmt="d")
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.savefig(f"results_eval/cm_images/{filename}", bbox_inches='tight')
    plt.close()

# Generate confusion matrices
plot_cm(data["Imbalanced"]["ConfusionMatrix"], "Imbalanced Dataset", "cm_imbalanced.png")
plot_cm(data["LocalDeepSMOTE"]["ConfusionMatrix"], "Local DeepSMOTE", "cm_local.png")
plot_cm(data["FedDeepSMOTE"]["ConfusionMatrix"], "FedDeepSMOTE", "cm_fed.png")

print("Confusion matrix images saved in results_eval/cm_images/")