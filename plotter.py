import csv
import numpy as np
import matplotlib.pyplot as plt

losses = np.load("./cosFace/checkpoint/loss.npy")
accuracies = np.load("./cosFace/checkpoint/accuracy.npy")
print(losses.shape, accuracies.shape)
file_name = "cos_casia"

data = accuracies
# data = losses

epochs = np.arange(1,data.shape[0]+1,1)
train_loss = data


plt.title("Accuracy Curve for Cos Face on CASIA-Webface Dataset")
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.plot(epochs, train_loss, color='r', label='Training Accuracy')
plt.grid(True)
plt.legend()
plt.savefig("./cosFace/results/%s_acc.png"%file_name)
plt.show()
