import csv
import numpy as np
import matplotlib.pyplot as plt

losses = np.load("./sphereFace/checkpoint/loss.npy")
accuracies = np.load("./sphereFace/checkpoint1/accuracy.npy")
print(losses.shape, accuracies.shape)
file_name = "sphere_casia"

# data = accuracies
data = losses

epochs = np.arange(1,data.shape[0]+1,1)
train_loss = data


plt.title("Loss Curve of Sphere Face on CASIA-Webface Dataset")
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.plot(epochs, train_loss, color='r', label='Training Loss')
plt.grid(True)
plt.legend()
plt.savefig("./sphereFace/results/%s_loss.png"%file_name)
plt.show()
