"""This script generates bar charts and line plots to visualize the 
performance of different data augmentation methods 
(Imbalanced, Local DeepSMOTE, FedDeepSMOTE) based on evaluation metrics such as F1-score, G-Mean, and ACSA. 
The generated plots are saved as PNG files for further analysis.
"""


import matplotlib.pyplot as plt
import numpy as np

# Data (average of your clients OR use one set)
methods = ["Imbalanced", "Local DeepSMOTE", "FedDeepSMOTE"]

f1 = [0.861, 0.951, 0.953]
gmean = [0.835, 0.950, 0.952]
acsa = [0.873, 0.952, 0.953]

x = np.arange(len(methods))
width = 0.25

plt.figure()

plt.bar(x - width, f1, width, label="F1-score")
plt.bar(x, gmean, width, label="G-Mean")
plt.bar(x + width, acsa, width, label="ACSA")

plt.xticks(x, methods)
plt.ylabel("Score")
plt.title("Performance Comparison of Data Augmentation Methods")

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("results_bar_chart.png")
plt.show()



improvement_f1 = [0.861, 0.951, 0.953]

plt.figure()

plt.plot(methods, improvement_f1, marker='o')
plt.ylabel("Macro F1-score")
plt.title("Performance Improvement Across Methods")

plt.grid(True)

plt.savefig("f1_trend.png")
plt.show()