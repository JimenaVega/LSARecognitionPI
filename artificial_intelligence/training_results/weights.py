import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv


SELECTED = 3

load_dotenv()

DATAPATH = os.getenv('DATAPATH')

with open(DATAPATH + 'artificial_intelligence/training_results/' + 'info.csv', mode='r') as training_info:
    file_rows = list(csv.reader(training_info))
    param_legends = file_rows[0][3:]
    file_rows = file_rows[1:]

    selected_row = file_rows[SELECTED]
    selected_logs = selected_row[2]
    param_values = selected_row[3:]
    pass

data = pd.read_csv(DATAPATH + 'artificial_intelligence/training_results/weights/' + f'{selected_logs}.csv')

metrics = ["categorical_accuracy", "val_categorical_accuracy", "loss", "val_loss"]

fig, ax = plt.subplots()

for column in metrics[0:2]:
    ax.plot(data["epoch"], data[column])

ax.set_xlabel("Epochs")
ax.set_ylabel("Metrics")
ax.set_title(f'Metrics evolution over epochs | {selected_logs} file')
ax.legend(metrics)

textstr = ''
for i, (parametro, valor) in enumerate(zip(param_legends, param_values)):
    if i == 0:
        textstr += f'{parametro}: {valor}'
    else:
        textstr += f'\n{parametro}: {valor}'


props = dict(boxstyle='round', facecolor='white', alpha=0.2)

# place a text box in upper left in axes coords
ax.text(0.01, 0.99, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.show()