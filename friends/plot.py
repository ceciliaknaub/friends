import matplotlib.pyplot as plt
import numpy as np

metrics = ['Baseline', 'LR']
pct = [0.18, 0.2873]

plt.bar(metrics, pct, color='blue')
plt.title('Dev Accuracy')
plt.xlabel('Model')
plt.ylabel('Pct')
plt.show()
"""