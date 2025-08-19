import matplotlib.pyplot as plt

models = ['CNN', 'LSTM', 'LightGBM', 'GMM']
accuracy = [30.54, 26.32, 73.35, 25.46]
f1 = [27.52, 27.25, 72.86, 26.42]

plt.bar(models, accuracy, alpha=0.6, label='Accuracy')
plt.bar(models, f1, alpha=0.6, label='Weighted F1')
plt.ylabel('Score (%)')
plt.title('Model Comparison with windowing')
plt.legend()
plt.show()