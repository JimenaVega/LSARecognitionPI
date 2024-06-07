import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("islr-fp16-192-8-seed42-fold0-logs.csv")

metrics = ["categorical_accuracy", "val_categorical_accuracy", "loss", "val_loss"]

fig, ax = plt.subplots()

for column in metrics[:2]:
    ax.plot(data["epoch"], data[column])

ax.set_xlabel("Epochs")
ax.set_ylabel("Metrics")
ax.set_title("Metrics evolution over epochs")
ax.legend(metrics)

plt.show()