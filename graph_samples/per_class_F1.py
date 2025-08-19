import matplotlib.pyplot as plt

import numpy as np
classes = ['Base', 'Aggression', 'Social', 'Nonsocial']
cnn_f1 = [18.14, 0.92, 43.09, 26.91]
lstm_f1 = [21.10, 15.42, 31.29, 30.59]
lgbm_f1 = [67.99, 62.12, 75.57, 74.51]
gmm_f1 = [21.08, 16.88, 29.78, 29.17]
bar_width = 0.2
x = np.arange(len(classes))
plt.bar(x - 1.5*bar_width, cnn_f1, width=bar_width, label='CNN')
plt.bar(x - 0.5*bar_width, lstm_f1, width=bar_width, label='LSTM')
plt.bar(x + 0.5*bar_width, lgbm_f1, width=bar_width, label='LightGBM')
plt.bar(x + 1.5*bar_width, gmm_f1, width=bar_width, label='GMM')
plt.xticks(x, classes)
plt.ylabel('F1-score')
plt.title('Per-Class F1-score by Model')
plt.legend()
plt.show()